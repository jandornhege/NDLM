#!/usr/bin/env python3
"""Analyze benchmark learners on LPs excluded from first-fraction selection.

The split is always defined from --selection-root (normally ALL), never from
an evaluation output root. LP names are sorted lexicographically per domain;
the first fraction is the selection set and the remainder is the evaluation
set. This guarantees that the two sets are disjoint when the same LP identity
is used in both roots.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
MULTI_LP_DOMAINS = (
    "Biopax",
    "Carcinogenesis_drill",
    "Family",
    "Mutagenesis_drill",
)

LEARNERS = ("ALCSAT", "CELOE", "Drill", "EvoLearner", "OCEL", "TDL")

NDLM_FOLD_PATTERN = re.compile(
    r"Fold\s+(?P<fold>\d+):\s*"
    r"TP=(?P<tp>\d+),\s*"
    r"FP=(?P<fp>\d+),\s*"
    r"TN=(?P<tn>\d+),\s*"
    r"FN=(?P<fn>\d+)"
)
LEARNER_F1_PATTERN = re.compile(
    r"Fold\s+(?P<fold>\d+):\s*"
    r"(?P<learner>ALCSAT|CELOE|Drill|EvoLearner|OCEL|TDL)\s+"
    r"F1=(?P<f1>\d+(?:\.\d+)?)"
)


def parse_summary(path: Path, root: Path) -> dict | None:
    relative = path.relative_to(root).parts
    if len(relative) != 6 or relative[-1] != "summary.log":
        return None
    configuration, domain, neighborhood_name, problem, run_name, _ = relative
    if not neighborhood_name.startswith("neighborhood_") or not run_name.startswith("run_"):
        return None
    try:
        neighborhood = int(neighborhood_name.removeprefix("neighborhood_"))
        run = int(run_name.removeprefix("run_"))
    except ValueError:
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    ndlm = {}
    for match in NDLM_FOLD_PATTERN.finditer(text):
        ndlm[int(match["fold"])] = {
            "tp": int(match["tp"]),
            "fp": int(match["fp"]),
            "tn": int(match["tn"]),
            "fn": int(match["fn"]),
        }

    learner_f1 = defaultdict(dict)
    for match in LEARNER_F1_PATTERN.finditer(text):
        learner_f1[match["learner"]][int(match["fold"])] = float(match["f1"])

    if not ndlm and not learner_f1:
        return None
    return {
        "configuration": configuration,
        "domain": domain,
        "neighborhood": neighborhood,
        "problem": problem,
        "run": run,
        "ndlm": ndlm,
        "learner_f1": dict(learner_f1),
        "path": str(path),
    }


def collect(root: Path, domain: str) -> list[dict]:
    grouped = {}
    for path in root.glob(f"*/{domain}/**/summary.log"):
        result = parse_summary(path, root)
        if result is None:
            continue
        key = (
            result["configuration"],
            result["domain"],
            result["neighborhood"],
            result["problem"],
        )
        current = grouped.get(key)
        rank = (len(result["ndlm"]), result["run"])
        if current is None or rank > (len(current["ndlm"]), current["run"]):
            grouped[key] = result
    return list(grouped.values())


def canonical_split(selection_root: Path, domain: str, fraction: float, seed: int):
    results = collect(selection_root, domain)
    problems = sorted({result["problem"] for result in results})
    random.Random(seed).shuffle(problems)
    split_index = max(1, int(len(problems) * fraction))
    return problems[:split_index], problems[split_index:]


def load_or_create_splits(selection_root, domains, fraction, seed, split_file):
    """Load a fixed split or create and persist it once."""
    if split_file.exists():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        if payload.get("fraction") != fraction or payload.get("seed") != seed:
            raise RuntimeError(
                f"Split file {split_file} was created with "
                f"fraction={payload.get('fraction')}, seed={payload.get('seed')}; "
                f"requested fraction={fraction}, seed={seed}"
            )
        return payload["domains"]

    split_data = {
        "fraction": fraction,
        "seed": seed,
        "domains": {},
    }
    for domain in domains:
        train, test = canonical_split(selection_root, domain, fraction, seed)
        split_data["domains"][domain] = {
            "train": train,
            "test": test,
        }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split_data, indent=2) + "\n", encoding="utf-8")
    return split_data["domains"]


def totals(folds):
    return {
        metric: sum(fold[metric] for fold in folds)
        for metric in ("tp", "fp", "tn", "fn")
    }


def f1_from_counts(counts):
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denominator if denominator else 0.0


def summarize(results, selected_last):
    selected = set(selected_last)
    grouped = defaultdict(list)
    for result in results:
        if result["problem"] in selected:
            grouped[(result["configuration"], result["neighborhood"])].append(result)

    output = {}
    for key, problem_results in grouped.items():
        ndlm_folds = [
            fold
            for result in problem_results
            for fold in result["ndlm"].values()
        ]
        ndlm_f1 = f1_from_counts(totals(ndlm_folds)) if ndlm_folds else None
        output[key] = {"NDLM": ndlm_f1}
        for learner in LEARNERS:
            f1_values = [
                value
                for result in problem_results
                for value in result["learner_f1"].get(learner, {}).values()
            ]
            output[key][learner] = sum(f1_values) / len(f1_values) if f1_values else None
    return output


def select_best_configuration(results, selected_first):
    """Select the best complete configuration using only selected LPs."""
    selected = set(selected_first)
    grouped = defaultdict(list)
    for result in results:
        if result["problem"] in selected:
            grouped[(result["configuration"], result["neighborhood"])].append(result)

    candidates = []
    for (configuration, neighborhood), problem_results in grouped.items():
        if len(problem_results) != len(selected_first):
            continue
        per_problem_f1 = [
            f1_from_counts(totals(result["ndlm"].values()))
            for result in problem_results
            if result["ndlm"]
        ]
        if len(per_problem_f1) != len(selected_first):
            continue
        folds = [
            fold
            for result in problem_results
            for fold in result["ndlm"].values()
        ]
        candidates.append(
            {
                "configuration": configuration,
                "neighborhood": neighborhood,
                "macro_f1": sum(per_problem_f1) / len(per_problem_f1),
                "micro_f1": f1_from_counts(totals(folds)),
            }
        )
    if not candidates:
        raise RuntimeError("No complete configuration covers the randomized selection LPs")
    return max(candidates, key=lambda candidate: candidate["macro_f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=Path("outputs/CONCEPT_LEARNER/ALL"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("outputs/CONCEPT_LEARNER/BENCHMARKS_BEST"))
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("outputs/CONCEPT_LEARNER/lp_split_seed42.json"),
        help="Persist and reuse the randomized LP split from this JSON file.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be greater than 0 and less than 1")

    split_domains = load_or_create_splits(
        args.selection_root,
        MULTI_LP_DOMAINS,
        args.fraction,
        args.seed,
        args.split_file,
    )
    report = {
        "fraction": args.fraction,
        "seed": args.seed,
        "split_file": str(args.split_file),
        "domains": {},
    }
    for domain in MULTI_LP_DOMAINS:
        selection_results = collect(args.selection_root, domain)
        first = split_domains[domain]["train"]
        last = split_domains[domain]["test"]
        evaluation_results = collect(args.evaluation_root, domain)
        evaluation_problems = {result["problem"] for result in evaluation_results}
        overlap = set(first) & set(last)
        missing = set(last) - evaluation_problems
        if overlap:
            raise RuntimeError(f"{domain}: first/last LP split overlaps: {sorted(overlap)}")
        if missing:
            raise RuntimeError(
                f"{domain}: evaluation root is missing {len(missing)} canonical last-fraction LPs"
            )

        selected_configuration = select_best_configuration(selection_results, first)
        values_by_config = summarize(evaluation_results, last)
        preferred_configuration = "L3H5_identity" if domain == "Biopax" else "BEST"
        preferred = {
            key: values
            for key, values in values_by_config.items()
            if key[0] == preferred_configuration
        }
        if len(preferred) != 1:
            raise RuntimeError(
                f"{domain}: expected one {preferred_configuration} configuration, found {list(preferred)}"
            )
        preferred_key, preferred_values = next(iter(preferred.items()))

        report["domains"][domain] = {
            "total_problems": len(first) + len(last),
            "first_fraction_problems": len(first),
            "last_fraction_problems": len(last),
            "split_overlap": len(overlap),
            "missing_last_fraction_problems": len(missing),
            "seed": args.seed,
            "selected_configuration": selected_configuration["configuration"],
            "selected_neighborhood": selected_configuration["neighborhood"],
            "selected_macro_f1": selected_configuration["macro_f1"],
            "selected_micro_f1": selected_configuration["micro_f1"],
            "configuration": preferred_key[0],
            "neighborhood": preferred_key[1],
            "scores": preferred_values,
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("Randomized holdout benchmark scores; first-fraction LPs are excluded")
    for domain, values in report["domains"].items():
        print(
            f"\n{domain}: selected={values['selected_configuration']}, "
            f"selected neighborhood={values['selected_neighborhood']}; "
            f"evaluated={values['configuration']}, "
            f"evaluation neighborhood={values['neighborhood']}"
        )
        print(
            f"LPs: first={values['first_fraction_problems']}, "
            f"last={values['last_fraction_problems']}, "
            f"overlap={values['split_overlap']} | "
            f"selection macro F1={values['selected_macro_f1'] * 100:.2f}"
        )
        for learner, score in sorted(values["scores"].items()):
            formatted = "N/A" if score is None else f"{score * 100:.2f}"
            print(f"  {learner}: {formatted}")


if __name__ == "__main__":
    main()
