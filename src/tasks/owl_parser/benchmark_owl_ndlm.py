"""Run NDLM inference benchmarks on exported OWL tensors and LP targets."""

import argparse
import json
from pathlib import Path
import resource
import time
from sklearn.model_selection import StratifiedKFold
import argparse
import os
import torch

from difflogic.nn.neural_logic import layer
from tasks.owl_parser.owl_tensor_parser import owl_to_tensors


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "src" / "data" / "Ontolearn"
DEFAULT_TENSORS = Path(__file__).with_name("ntn_tensors.pt")


def current_rss_bytes() -> int:
    """Read the current resident set size on Linux."""
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS was not available in /proc/self/status")


def normalize_rdf_term(value: str) -> str:
    """Normalize a graph term from an OWL tensor payload or LP JSON file."""
    text = str(value).strip()
    if not text:
        return text
    if text.startswith("<") and text.endswith(">"):
        return text
    if text.startswith("http://") or text.startswith("https://"):
        return f"<{text}>"
    return text


def choose_tensor_for_kg(kg_dir: Path) -> Path | None:
    """Prefer the standard ontology export over an L3 variant when available."""
    kg_files = sorted(kg_dir.rglob("*.pt"))
    if not kg_files:
        return None
    non_l3 = [
        path
        for path in kg_files
        if "l3" not in path.name.lower() and "level3" not in path.name.lower()
    ]
    return non_l3[0] if non_l3 else kg_files[0]


def discover_working_problems(data_root: Path) -> list[tuple[Path, Path]]:
    """Find every KG tensor/LP JSON pair that has an exported ontology tensor."""
    kg_root = data_root / "KGs"
    lp_root = data_root / "LPs"
    pairs: list[tuple[Path, Path]] = []
    if not kg_root.is_dir() or not lp_root.is_dir():
        return pairs

    for kg_dir in sorted(kg_root.iterdir()):
        if not kg_dir.is_dir():
            continue
        tensor_path = choose_tensor_for_kg(kg_dir)
        if tensor_path is None:
            continue
        lp_dir = lp_root / kg_dir.name
        if not lp_dir.is_dir():
            continue
        for lp_path in sorted(lp_dir.glob("*.json")):
            pairs.append((tensor_path, lp_path))
    return pairs


def iter_lp_problems(raw):
    """Normalize Ontolearn LP structures into problem dictionaries."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        problems = raw.get("problems")
        if isinstance(problems, dict):
            return [
                {"target expression": name, "examples": examples}
                for name, examples in problems.items()
            ]
        if isinstance(problems, list):
            return problems
        if "target expression" in raw or "examples" in raw:
            return [raw]
    return []


def load_lp_target(payload: dict, lp_path: Path):
    """Load a single LP JSON file into tensors with a positive/negative mask."""
    with lp_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    problems = iter_lp_problems(raw)

    object_index = payload["object_index"]
    num_objects = payload["concepts"].shape[1]
    targets = []
    for problem_index, problem in enumerate(problems):
        if not isinstance(problem, dict):
            continue
        examples = problem.get("examples", {})
        pos_values = examples.get("positive_examples", examples.get("positive examples", []))
        neg_values = examples.get("negative_examples", examples.get("negative examples", []))

        target = torch.zeros(num_objects, dtype=torch.float32)
        mask = torch.zeros(num_objects, dtype=torch.bool)
        num_pos = len(pos_values)
        num_neg = len(neg_values)
        for term in pos_values:
            key = normalize_rdf_term(term)
            if key in object_index:
                index = object_index[key]
                target[index] = 1.0
                mask[index] = True
        for term in neg_values:
            key = normalize_rdf_term(term)
            if key in object_index:
                index = object_index[key]
                target[index] = 0
                mask[index] = True

        target = torch.where(mask, target, torch.zeros_like(target)).float()
        targets.append(
            {
                "name": problem.get("target expression", f"{lp_path.stem}_{problem_index}"),
                "target": target,
                "mask": mask,
                "num_positive": num_pos,
                "num_negative": num_neg,
            }
        )
    return targets

def load_lp_examples(lp_path: Path):
    with lp_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    problems = iter_lp_problems(raw)

    res = {}

    for problem_index, problem in enumerate(problems):
        examples = []
        if not isinstance(problem, dict):
            continue

        lp_name = problem.get(
            "target expression",
            f"{lp_path.stem}_{problem_index}",
        )

        problem_examples = problem.get("examples", {})

        positives = problem_examples.get(
            "positive_examples",
            problem_examples.get("positive examples", []),
        )

        negatives = problem_examples.get(
            "negative_examples",
            problem_examples.get("negative examples", []),
        )

        for i, example in enumerate(positives):
            examples.append({
                "name": f"{lp_name}_positive_{i}",
                "center": normalize_rdf_term(example),
                "label": 1,
            })

        for i, example in enumerate(negatives):
            examples.append({
                "name": f"{lp_name}_negative_{i}",
                "center": normalize_rdf_term(example),
                "label": 0,
            })
        res[lp_name] = examples

    return res

def make_learning_problem(
    owl_files,
    center,
    label,
    *,
    name=None,
    radius=0,
    mark_target_object=False,
    mark_target_object_keep_mask=False,
):
    """
    Create one learning problem centered on a single example.

    The input graph is restricted to the local neighborhood of `center`.
    The target/mask contain exactly one supervised object: `center`, unless
    `mark_target_object` is set and `mark_target_object_keep_mask` is not,
    in which case every object in the neighborhood is supervised.

    Returned tensor shapes (`num_objects` = number of objects in the local
    neighborhood, `num_concepts` = number of concept names in the ontology):
      - `concepts`: (num_concepts, num_objects) float, or
        (num_concepts + 1, num_objects) if `mark_target_object` is set, with
        the extra last row one-hot at `center_index`.
      - `roles`: (num_roles, num_objects, num_objects) float.
      - `target`/`mask`: (num_objects,) float/bool, both fully populated when
        `mark_target_object` is set without `mark_target_object_keep_mask`,
        otherwise only set at `center_index`.
      - `role_target`/`role_mask`: (0, num_objects, num_objects), unused.
    """

    # Generate the local KG around this example
    payload = owl_to_tensors(
        owl_files,
        neighborhood_node=center,
        neighborhood_distance=radius,
    )

    # The center must be present in the local graph
    if center not in payload.object_index:
        raise ValueError(
            f"Center {center!r} was not found in the local tensor."
        )

    center_index = payload.object_index[center]
    num_objects = payload.concepts.shape[1]
    concepts = payload.concepts

    if mark_target_object:
        target_concept = torch.zeros(
            (1, num_objects),
            dtype=payload.concepts.dtype,
        )
        target_concept[0, center_index] = 1.0
        concepts = torch.cat((payload.concepts, target_concept), dim=0)

    target = torch.zeros(
        num_objects,
        dtype=torch.float32,
    )

    mask = torch.zeros(
        num_objects,
        dtype=torch.bool,
    )

    if mark_target_object and not mark_target_object_keep_mask:
        target.fill_(float(label))
        mask.fill_(True)
    else:
        target[center_index] = float(label)
        mask[center_index] = True

    # No role learning
    role_target = torch.zeros(
        (0, num_objects, num_objects),
        dtype=torch.float32,
    )

    role_mask = torch.zeros(
        (0, num_objects, num_objects),
        dtype=torch.bool,
    )

    return {
        "name": name or str(center),
        "center": center,
        "label": int(label),
        "center_index": center_index,

        "concepts": concepts,
        "roles": payload.roles,

        "target": target,
        "mask": mask,

        "role_target": role_target,
        "role_mask": role_mask,

        # Keep the metadata in case you need it later
        "object_terms": payload.object_terms,
        "concept_names": payload.concept_names,
        "role_names": payload.role_names,
        "object_index": payload.object_index,
    }


def benchmark_single_inference(
    tensor_path: Path,
    *,
    max_objects: int,
    device: torch.device,
    project_concepts: int,
    project_roles: int,
    nlm_breadth: int = 3,
    lp_path: Path | None = None,
) -> dict:
    """Run a single NDLM forward pass and return its runtime and memory stats."""
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    concepts = payload["concepts"]
    roles = payload["roles"]
    total_objects = concepts.shape[1]
    num_objects = total_objects if max_objects == 0 else min(max_objects, total_objects)
    concepts = concepts[:, :num_objects].unsqueeze(0).to(device)
    roles = roles[:, :num_objects, :num_objects].unsqueeze(0).to(device)

    if project_concepts <= 0 or project_roles <= 0:
        raise ValueError("NLM_to_NDLM_Adapter requires positive hidden concept and role sizes")
    if nlm_breadth < 3:
        raise ValueError("NLM_to_NDLM_Adapter requires nlm_breadth >= 3")

    adapter_args = argparse.Namespace(
        nlm_breadth=nlm_breadth,
        hidden_concepts=project_concepts,
        hidden_roles=project_roles,
        activation_function="identity",
        nlm_exclude_self=False,
        nlm_residual=False,
        num_layers=1,
    )

    model = layer.NLM_to_NDLM_Adapter(
        concepts.shape[1],
        roles.shape[1],
        10,
        10,
        adapter_args,
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rss_before = current_rss_bytes()
    start = time.perf_counter()
    with torch.inference_mode():
        output_concepts, output_roles = model(concepts, roles)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    rss_after = current_rss_bytes()
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    lp_targets = load_lp_target(payload, lp_path) if lp_path is not None else []
    problem_summary = []
    for target in lp_targets:
        problem_summary.append(
            {
                "target_name": target["name"],
                "positive_examples": int(target["positive"].sum().item()),
                "negative_examples": int(target["negative"].sum().item()),
                "mask_unique": sorted(torch.unique(target["mask"]).tolist()),
            }
        )

    return {
        "tensor_path": str(tensor_path),
        "lp_path": str(lp_path) if lp_path is not None else None,
        "device": str(device),
        "num_objects": num_objects,
        "total_objects": total_objects,
        "input_concepts_shape": tuple(concepts.shape),
        "input_roles_shape": tuple(roles.shape),
        "ndlm_input_concepts": (1, project_concepts, num_objects),
        "ndlm_input_roles": (1, project_roles, num_objects, num_objects),
        "output_concepts_shape": tuple(output_concepts.shape),
        "output_roles_shape": tuple(output_roles.shape),
        "inference_time_seconds": elapsed_seconds,
        "cpu_rss_before_mib": rss_before / 2**20,
        "cpu_rss_after_mib": rss_after / 2**20,
        "cpu_peak_rss_mib": peak_rss_bytes / 2**20,
        "peak_cuda_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
        "lp_targets": problem_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors", type=Path, default=DEFAULT_TENSORS, help="Single tensor file to benchmark.")
    parser.add_argument("--lp", type=Path, default=None, help="LP JSON file to load target masks for the chosen tensor.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=64,
        help="Number of objects passed to NDLM; use 0 to keep the full graph.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--project-concepts", type=int, default=10, help="Projected hidden concept width; use 0 to disable projection.")
    parser.add_argument("--project-roles", type=int, default=10, help="Projected hidden role width; use 0 to disable projection.")
    parser.add_argument("--nlm-breadth", type=int, default=3, help="NLM maximum arity/breadth; learn_task.py uses 3.")
    parser.add_argument("--all", action="store_true", help="Benchmark all exported KG tensors with matching LP JSON files.")
    args = parser.parse_args()

    if args.max_objects < 0:
        parser.error("--max-objects must be non-negative")
    if args.project_concepts < 0 or args.project_roles < 0:
        parser.error("projected concept and role counts must both be non-negative")
    if args.nlm_breadth < 3:
        parser.error("--nlm-breadth must be at least 3")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.all:
        targets = discover_working_problems(args.data_root)
        if not targets:
            raise FileNotFoundError(f"no exported KG tensor / LP pairs were found under {args.data_root}")
        for tensor_path, lp_path in targets:
            summary = benchmark_single_inference(
                tensor_path,
                max_objects=args.max_objects,
                device=device,
                project_concepts=args.project_concepts,
                project_roles=args.project_roles,
                nlm_breadth=args.nlm_breadth,
                lp_path=lp_path,
            )
            print(f"tensor: {summary['tensor_path']}")
            print(f"lp: {summary['lp_path']}")
            print(f"device: {summary['device']}")
            print(f"objects: {summary['num_objects']}/{summary['total_objects']}")
            print(f"nlm hidden concepts: {summary['ndlm_input_concepts']}")
            print(f"nlm hidden roles: {summary['ndlm_input_roles']}")
            print(f"output concepts: {summary['output_concepts_shape']}")
            print(f"output roles: {summary['output_roles_shape']}")
            print(f"inference time: {summary['inference_time_seconds']:.3f} s")
            print(f"CPU RSS before/after: {summary['cpu_rss_before_mib']:.1f}/{summary['cpu_rss_after_mib']:.1f} MiB")
            print(f"CPU peak RSS: {summary['cpu_peak_rss_mib']:.1f} MiB")
            if device.type == "cuda":
                print(f"CUDA peak allocated: {summary['peak_cuda_allocated_mib']:.1f} MiB")
            if summary["lp_targets"]:
                for target in summary["lp_targets"]:
                    print(
                        "target output: "
                        f"{target['target_name']}, "
                        f"positive={target['positive_examples']}, "
                        f"negative={target['negative_examples']}, "
                        f"mask_unique={target['mask_unique']}"
                    )
            print("-" * 72)
        return

    if not args.tensors.exists():
        raise FileNotFoundError(f"tensor file not found: {args.tensors}")
    summary = benchmark_single_inference(
        args.tensors,
        max_objects=args.max_objects,
        device=device,
        project_concepts=args.project_concepts,
        project_roles=args.project_roles,
        nlm_breadth=args.nlm_breadth,
        lp_path=args.lp,
    )
    print(f"tensor: {summary['tensor_path']}")
    if summary['lp_path'] is not None:
        print(f"lp: {summary['lp_path']}")
    print(f"device: {summary['device']}")
    print(f"objects: {summary['num_objects']}/{summary['total_objects']}")
    print(f"input concepts: {summary['input_concepts_shape']}")
    print(f"input roles: {summary['input_roles_shape']}")
    print(f"NLM hidden concepts: {summary['ndlm_input_concepts']}")
    print(f"NLM hidden roles: {summary['ndlm_input_roles']}")
    print(f"output concepts: {summary['output_concepts_shape']}")
    print(f"output roles: {summary['output_roles_shape']}")
    print(f"inference time: {summary['inference_time_seconds']:.3f} s")
    print(f"CPU RSS before/after: {summary['cpu_rss_before_mib']:.1f}/{summary['cpu_rss_after_mib']:.1f} MiB")
    print(f"CPU peak RSS: {summary['cpu_peak_rss_mib']:.1f} MiB")
    if device.type == "cuda":
        print(f"CUDA peak allocated: {summary['peak_cuda_allocated_mib']:.1f} MiB")
    for target in summary["lp_targets"]:
        print(
            "target output: "
            f"{target['target_name']}, "
            f"positive={target['positive_examples']}, "
            f"negative={target['negative_examples']}, "
            f"mask_unique={target['mask_unique']}"
        )


if __name__ == "__main__":
    # main()
   
    lp_path= Path("src/data/Ontolearn/LPs/Family/lps_difficult.json")
    # Load the examples produced by Step 2
    examples = load_lp_examples(lp_path)

    # Pick one example
    example = examples[0]

    print("Example:")
    print(example)

    # Create its local learning problem
    problem = make_learning_problem(
        owl_files=(
            "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.owl",
            # "src/data/Ontolearn/KGs/Family/father.owl"
        ),
        center=example["center"],
        label=example["label"],
        name=example["name"],

        radius=5,
    )
    print(problem["concept_names"])
    print(problem["role_names"])

    print("\nLearning problem:")
    print("name:", problem["name"])
    print("center:", problem["center"])
    print("label:", problem["label"])

    print("\nTensor shapes:")
    print("concepts:", problem["concepts"].shape)
    print("roles:", problem["roles"].shape)

    print("\nCenter:")
    print("center index:", problem["center_index"])
    print(
        "center term:",
        problem["object_terms"][problem["center_index"]],
    )

    print("\nTarget:")
    print(problem["target"])

    print("\nMask:")
    print(problem["mask"])

    # Sanity checks
    assert problem["mask"].sum().item() == 1
    assert problem["mask"][problem["center_index"]]
    assert problem["target"][problem["center_index"]].item() == example["label"]

    print("\nNumber of local objects:", problem["concepts"].shape[1])
    print("Number of supervised objects:", problem["mask"].sum().item())