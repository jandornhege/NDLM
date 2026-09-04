"""Train one NDLM per learning problem on *all* examples (no cross validation)
and prune the resulting model afterwards."""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from ndlm.configs import config_object
from ndlm.modules import MultiLayerNDLM
import ndlm.main as NDLM_main
import owl_parser.benchmark_owl_ndlm as benchmark_owl_ndlm
from Pruning import (
    model_to_verbal_description,
    plot_weight_distribution,
    prune_weights_batchwise,
    save_model,
)

from my_logging import log, init_logger

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = {
    "NTN": "src/tasks/owl_parser/ntn_entities.pt",
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.entities.pt",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.entities.pt",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.entities.pt"
}
owl_files = {
    "NTN": "src/tasks/owl_parser/ntn_entities.pt",
    "Biopax": "src/data/Ontolearn/KGs/Biopax/biopax.entities.pt",
    "Family": "src/data/Ontolearn/KGs/Family/family-benchmark_rich_background.owl",
    "Lymphography": "src/data/Ontolearn/KGs/Lymphography/lymphography.entities.pt"
}

lp_files = {
    "Biopax": "src/data/Ontolearn/LPs/Biopax/lps.json",
    "Family": "src/data/Ontolearn/LPs/Family/lps_difficult.json",
    "Lymphography": "src/data/Ontolearn/LPs/Lymphography/lps.json"
}


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--neighborhood", type=int, default=4)
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--hidden-concepts", type=int, default=5)
parser.add_argument("--hidden-roles", type=int, default=5)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-4,
    help="Adam weight decay; default preserves the original 1e-4 setting.",
)
parser.add_argument(
    "--activation",
    choices=("identity", "sigmoid"),
    default="sigmoid",
)
parser.add_argument("--configuration-name", default="L3H5_sig")
parser.add_argument(
    "--output-root",
    default="FULL_TRAIN_PRUNE",
    help="Directory below outputs/CONCEPT_LEARNER for this experiment round.",
)
parser.add_argument(
    "--problem-index",
    type=int,
    default=None,
    help="Zero-based index of one learning problem to run; omit to run all.",
)
parser.add_argument(
    "--mark-target-object",
    action="store_true",
    help=(
        "Add a concept marking the target object and supervise the target "
        "for every object in the local neighborhood."
    ),
)
parser.add_argument(
    "--num-epochs",
    type=int,
    default=500,
)
parser.add_argument(
    "--train-time-limit-seconds",
    type=float,
    default=None,
    help=(
        "Optional strict wall-clock limit for training; training stops early "
        "(after one final test round) once this many seconds have elapsed."
    ),
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help=(
        "Log detailed per-dataset metrics every epoch (large log files). "
        "By default only one summary line per epoch/test round is logged."
    ),
)
parser.add_argument(
    "--lr-scheduler",
    choices=("none", "plateau", "cosine"),
    default="none",
    help=(
        "Learning-rate schedule. 'plateau' halves the LR when the epoch "
        "training loss stops improving for 20 epochs; 'cosine' decays the "
        "LR smoothly to 0 over --num-epochs. Default 'none' keeps the "
        "original fixed learning rate."
    ),
)
parser.add_argument(
    "--prune-rounds",
    type=int,
    default=6,
    help=(
        "Maximum number of pruning passes; stops early once a pass removes "
        "no further weights."
    ),
)
parser.add_argument(
    "--skip-pruning",
    action="store_true",
    help="Only train and save the model, do not run the pruning stage.",
)
args = parser.parse_args()
name = args.name
tensor_path = data[name]
owl_files = owl_files[name]

lp_path = Path(lp_files[name])


experiment_path = (
    Path("outputs/CONCEPT_LEARNER")
    / args.output_root
    / args.configuration_name
    / name
    / f"neighborhood_{args.neighborhood}"
)

problems = benchmark_owl_ndlm.load_lp_examples(lp_path)
problem_items = list(problems.items())

if args.problem_index is not None:
    if not 0 <= args.problem_index < len(problem_items):
        raise ValueError(
            f"problem-index must be between 0 and {len(problem_items) - 1}, "
            f"got {args.problem_index}"
        )
    problem_items = [problem_items[args.problem_index]]

sample_cache = {}
name_cache = {}


def example_to_ndlm_sample(example, device):
    """Create an NDLM sample for one specific example in the local graph."""

    cache_key = example["name"]
    if cache_key not in sample_cache:
        problem = benchmark_owl_ndlm.make_learning_problem(
            owl_files=owl_files,
            center=example["center"],
            label=example["label"],
            name=example["name"],
            radius=args.neighborhood,
            mark_target_object=args.mark_target_object,
        )
        num_objects = problem["concepts"].shape[1]
        concept_names = list(problem["concept_names"])
        if args.mark_target_object:
            concept_names.append("target_marker")
        name_cache[cache_key] = {
            "c": concept_names,
            "r": list(problem["role_names"]),
            "c_out": ["target"],
            "r_out": [],
        }
        sample_cache[cache_key] = (
            problem["concepts"].unsqueeze(0),
            problem["roles"].unsqueeze(0),
            problem["target"].unsqueeze(0).unsqueeze(0),
            torch.zeros(
                (1, 0, num_objects, num_objects),
                dtype=torch.float32,
            ),
            problem["mask"].unsqueeze(0).unsqueeze(0),
            torch.zeros(
                (1, 0, num_objects, num_objects),
                dtype=torch.bool,
            ),
        )

    return tuple(tensor.to(device) for tensor in sample_cache[cache_key])


def count_misclassifications(model, dataset):
    """Masked misclassification counts of concept and role predictions."""

    concept_errors = 0
    role_errors = 0

    model.eval()
    with torch.no_grad():
        for concepts, roles, c_targets, r_targets, c_mask, r_mask in dataset:
            out_concepts, out_roles = model(concepts, roles)

            concept_errors += (
                ((out_concepts > 0) != (c_targets > 0)) & c_mask
            ).sum().item()

            if r_targets.numel() > 0:
                role_errors += (
                    ((out_roles > 0) != (r_targets > 0)) & r_mask
                ).sum().item()

    return concept_errors, role_errors


def make_evaluator(dataset):
    def evaluate(model):
        concept_errors, role_errors = count_misclassifications(model, dataset)
        res = {
            "train_c_changes": concept_errors,
            "train_r_changes": role_errors,
        }
        return concept_errors + role_errors, res

    return evaluate


def count_nonzero_weights(model):
    nonzero = 0
    total = 0
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            nonzero += (param != 0).sum().item()
            total += param.numel()
    return nonzero, total


def write_verbal_description(model, names, problem_path, lp_name, suffix):
    description, concepts, roles = model_to_verbal_description(
        model,
        lp_name,
        names=names,
    )

    description_path = problem_path / "model_descriptions"
    os.makedirs(description_path, exist_ok=True)

    with open(description_path / f"description_{lp_name}_{suffix}.txt", "w") as f:
        for key, value in description.items():
            f.write(f"{key}: {value}\n")

    with open(description_path / f"concept_names_{lp_name}_{suffix}.txt", "w") as f:
        for layer_names in concepts:
            f.write(f"{layer_names}\n")

    with open(description_path / f"role_names_{lp_name}_{suffix}.txt", "w") as f:
        for layer_names in roles:
            f.write(f"{layer_names}\n")

    return description


for lp_name, examples in problem_items:
    problem_path = experiment_path / lp_name / f"run_{time.time_ns()}"
    summary_log = problem_path / "summary.log"
    init_logger(summary_log)

    log(f"\n{'#' * 70}")
    log(f"Learning problem: {lp_name}")
    log(f"Number of examples: {len(examples)}")
    log(f"{'#' * 70}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train = [
        example_to_ndlm_sample(example, device)
        for example in examples
    ]

    log(f"{len(train)} many train datasets found with following shapes:")
    for i, ds in enumerate(train):
        log(
            f"Dataset {i}: concepts: {ds[0].shape}, roles: {ds[1].shape}, "
            f"c_target: {ds[2].shape}, r_target: {ds[3].shape} "
        )

    # ----------------------------------------------------
    # Configuration
    # ----------------------------------------------------

    config = config_object()

    config.NUM_LAYERS = args.num_layers
    config.NUM_HIDDEN_CONCEPTS = args.hidden_concepts
    config.NUM_HIDDEN_ROLES = args.hidden_roles
    config.MODE = "strict"
    config.TRANSITIVE_CLOSURE = False
    config.INITIAL_FFN = False
    config.ACTIVATION_FUNCTION = (
        nn.Sigmoid() if args.activation == "sigmoid" else nn.Identity()
    )

    training_args = argparse.Namespace(
        learning_rate=1e-3,
        weight_decay=args.weight_decay,
        batch_size=5,
        loss_type="BCE",
        num_epochs=args.num_epochs,
        test_interval=100,
        weighted_loss=False,
        experiment_path=problem_path,
        fold_time_limit_seconds=args.train_time_limit_seconds,
        verbose=args.verbose,
        lr_scheduler=args.lr_scheduler,
    )

    os.makedirs(training_args.experiment_path, exist_ok=True)

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    first_problem = train[0]

    model = MultiLayerNDLM(
        first_problem[0].shape[1],  # concepts, including optional marker
        first_problem[1].shape[1],  # roles
        1,
        0,
        config,
    ).to(device).eval()

    # ----------------------------------------------------
    # Train on all examples (train set is also used for reporting)
    # ----------------------------------------------------
    init_logger(problem_path / "training.log")

    tp, tn, fp, fn, _, _, _, _ = NDLM_main.main(
        train,
        train,
        config,
        training_args,
        model,
        checkpoint_path=training_args.experiment_path / "checkpoints",
        log=log,
    )
    init_logger(summary_log)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + fp + tn + fn)
        if tp + fp + tn + fn > 0
        else 0.0
    )

    log(
        f"{lp_name} — training fit: TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
        f"Precision={precision:.4f}, Recall={recall:.4f}, "
        f"F1={f1:.4f}, Accuracy={accuracy:.4f}"
    )

    plot_weight_distribution(training_args, model, lp_name)
    save_model(model, training_args, f"{lp_name}_original")

    if args.skip_pruning:
        continue

    # ----------------------------------------------------
    # Pruning
    # ----------------------------------------------------
    evaluate = make_evaluator(train)

    score, res = evaluate(model)
    nonzero, total = count_nonzero_weights(model)
    log(
        f"{lp_name} — before pruning: misclassifications={score} {res}, "
        f"non-zero weights={nonzero}/{total}"
    )

    # Each round starts with a fresh candidate list, so weights kept in an
    # earlier round can still be removed later.
    for prune_round in range(1, args.prune_rounds + 1):
        init_logger(problem_path / "pruning.log")
        model = prune_weights_batchwise(
            model,
            evaluate,
            training_args,
            f"{lp_name}_round_{prune_round}",
            verbose=True,
        )
        init_logger(summary_log)

        save_model(model, training_args, f"{lp_name}_pruned_round_{prune_round}")

        score, res = evaluate(model)
        previous_nonzero = nonzero
        nonzero, total = count_nonzero_weights(model)
        log(
            f"{lp_name} — after pruning round {prune_round}: "
            f"misclassifications={score} {res}, "
            f"non-zero weights={nonzero}/{total}"
        )

        if nonzero == previous_nonzero:
            log(f"{lp_name} — pruning converged after round {prune_round}.")
            break

    save_model(model, training_args, f"{lp_name}_pruned")

    description = write_verbal_description(
        model,
        name_cache[examples[0]["name"]],
        problem_path,
        lp_name,
        "pruned",
    )
    log(f"{lp_name} — verbal description of the pruned model:")
    for key, value in description.items():
        log(f"  {key}: {value}")
