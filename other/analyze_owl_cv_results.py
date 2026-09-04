#!/usr/bin/env python3
"""Collect F1 scores and per-fold confusion counts from OWL CV summaries."""

import argparse
import json
import re
from pathlib import Path
names=("Biopax",
    "Carcinogenesis",
    "Carcinogenesis_drill",
    "Family",
    "Lymphography",
    "Mutagenesis",
    "Mutagenesis_drill",
    "Nctrer",
    "Suramin"
)
DOMAIN_NAME = "Biopax"

FOLD_PATTERN = re.compile(
    r"Fold\s+(?P<fold>\d+):\s*"
    r"TP=(?P<tp>\d+),\s*"
    r"FP=(?P<fp>\d+),\s*"
    r"TN=(?P<tn>\d+),\s*"
    r"FN=(?P<fn>\d+)"
)
METHOD_F1_PATTERN = re.compile(
    r"Fold\s+(?P<fold>\d+):\s*"
    r"(?P<method>[A-Za-z0-9_-]+)\s+F1=(?P<f1>\d+(?:\.\d+)?)"
)
LEGACY_SWAPPED_METRICS = False


def fold_f1(fold: dict) -> float:
    """F1 of a single fold's confusion counts (0.0 when undefined)."""
    denominator = 2 * fold["tp"] + fold["fp"] + fold["fn"]
    return 2 * fold["tp"] / denominator if denominator else 0.0


def parse_summary(summary_path: Path, root: Path, check_benchmarks: bool) -> dict | None:
    """Parse one summary and return None for paths outside the expected layout."""
    relative_parts = summary_path.relative_to(root).parts
    if len(relative_parts) != 6:
        return None

    configuration, domain, neighborhood_name, problem, run_name, filename = relative_parts
    if filename != "summary.log" or not neighborhood_name.startswith("neighborhood_"):
        return None
    if not run_name.startswith("run_"):
        return None

    neighborhood = neighborhood_name.removeprefix("neighborhood_")
    try:
        neighborhood = int(neighborhood)
    except ValueError:
        return None

    text = summary_path.read_text(encoding="utf-8", errors="replace")
    folds = {}
    for match in FOLD_PATTERN.finditer(text):
        fp = int(match["tn"]) if LEGACY_SWAPPED_METRICS else int(match["fp"])
        tn = int(match["fp"]) if LEGACY_SWAPPED_METRICS else int(match["tn"])
        folds[int(match["fold"])] = {
            "tp": int(match["tp"]),
            "fp": fp,
            "tn": tn,
            "fn": int(match["fn"]),
        }

    
    # Parse CELOE / OCEL fold-level F1 scores from the new log format.
    # Parse fold-level F1 scores for all methods from the log.
    method_f1 = {}
    if check_benchmarks:
        for match in METHOD_F1_PATTERN.finditer(text):
            fold = int(match["fold"])
            method = match["method"]
            method_f1.setdefault(method, {})[fold] = float(match["f1"])


    # Keep the existing NDLM/model F1 calculation from confusion counts.
    f1 = None

    if folds:
        total_tp = sum(fold["tp"] for fold in folds.values())
        total_fp = sum(fold["fp"] for fold in folds.values())
        total_fn = sum(fold["fn"] for fold in folds.values())

        denominator = 2 * total_tp + total_fp + total_fn
        f1 = 2 * total_tp / denominator if denominator else 0.0

    # if domain != DOMAIN_NAME:
    #     return None

    return {

        "configuration": configuration,
        "domain": domain,
        "neighborhood": neighborhood,
        "problem": problem,
        "run": run_name.removeprefix("run_"),
        "f1": f1,
        "method_f1": {
            method: [
                {"fold": fold, "f1": f1}
                for fold, f1 in sorted(folds.items())
            ]
            for method, folds in method_f1.items()
        },
        "folds": [
            {"fold": fold, **folds[fold]}
            for fold in sorted(folds)
        ],
        "summary": str(summary_path),
    }


def collect_results(root: Path, check_benchmarks: bool, domain_name) -> list[dict]:
    DOMAIN_NAME = domain_name
    results = []
    for summary_path in root.glob(f"*/{DOMAIN_NAME}/**/summary.log"):
        result = parse_summary(summary_path, root, check_benchmarks=check_benchmarks)
        if result is not None and (result["f1"] is not None or result["folds"]):
            results.append(result)
    return keep_best_run_per_problem(results)


def keep_best_run_per_problem(results: list[dict]) -> list[dict]:
    """Keep only the best run_ folder per (configuration, domain,
    neighborhood, problem), instead of summing every rerun together (which
    would double-count folds and let crashed/partial runs masquerade as
    "complete"). Best = most completed folds, ties broken by the latest
    run timestamp.
    """
    best_by_key = {}
    for result in results:
        key = (
            result["configuration"],
            result["domain"],
            result["neighborhood"],
            result["problem"],
        )
        current_best = best_by_key.get(key)
        candidate_rank = (len(result["folds"]), int(result["run"]))
        if current_best is None or candidate_rank > (
            len(current_best["folds"]),
            int(current_best["run"]),
        ):
            best_by_key[key] = result
    return list(best_by_key.values())



def print_text(results: list[dict], check_benchmarks: bool) -> None:
    if not results:
        print("No fold or F1 results found.")
        return

    # for result in results:
    #     f1 = "not available" if result["f1"] is None else f"{result['f1']:.4f}"
    #     fold_values = "; ".join(
    #         f"F{fold['fold']}: TP={fold['tp']}, FP={fold['fp']}, "
    #         f"TN={fold['tn']}, FN={fold['fn']}"
    #         for fold in result["folds"]
    #     )
    #     print(
    #         f"{result['configuration']} | {result['domain']} | "
    #         f"neighborhood={result['neighborhood']} | problem={result['problem']} | "
    #         f"run={result['run']}"
    #     )
    #     print(f"  F1: {f1}")
    #     print(f"  folds: [{fold_values}]")

    overview = {}
    for result in results:
        setup = (result["configuration"], result["neighborhood"])
        summary = overview.setdefault(
            setup,
            {"runs": 0, "folds": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
        )
        summary["runs"] += 1
        for fold in result["folds"]:
            summary["folds"] += 1
            for metric in ("tp", "fp", "tn", "fn"):
                summary[metric] += fold[metric]

    overview_rows = []
    for (configuration, neighborhood), summary in overview.items():
        denominator = 2 * summary["tp"] + summary["fp"] + summary["fn"]
        f1 = 2 * summary["tp"] / denominator if denominator else 0.0
        overview_rows.append(
            (f1, configuration, neighborhood, summary)
        )

    print("\nRanked experiment/neighborhood combinations")
    for rank, (f1, configuration, neighborhood, summary) in enumerate(
        sorted(overview_rows, key=lambda row: row[0], reverse=True),
        start=1,
    ):
        complete_runs = sum(
            1
            for result in results
            if result["configuration"] == configuration
            and result["neighborhood"] == neighborhood
            and result["f1"] is not None
        )
        print(
            f"{rank}. {configuration} | neighborhood={neighborhood} | "
            f"combined F1={f1:.4f} | "
            f"complete runs={complete_runs}/{summary['runs']} | "
            f"folds={summary['folds']}"
        )
    # print("\nOverview by experiment setup and neighborhood")
    # for f1, configuration, neighborhood, summary in sorted(
    #     overview_rows,
    #     key=lambda row: row[0],
    #     reverse=True,
    # ):
    #     print(
    #         f"{configuration} | neighborhood={neighborhood}: "
    #         f"combined F1={f1:.4f} | "
    #         f"runs={summary['runs']}, folds={summary['folds']} | "
    #         f"TP={summary['tp']}, FP={summary['fp']}, "
    #         f"TN={summary['tn']}, FN={summary['fn']}"
    #     )

    # Macro aggregation: fold-level F1 averaged per problem, then averaged
    # across problems, matching how concept-learning papers report F1.
    if check_benchmarks:
        problem_method_f1s = {}

        problem_fold_f1s = {}

        for result in results:
            key = (
                result["configuration"],
                result["domain"],
                result["problem"],
                result["neighborhood"],
            )

            # NDLM F1 from confusion matrices
            problem_fold_f1s.setdefault(key, []).extend(
                fold_f1(fold) for fold in result["folds"]
            )

            # CELOE / OCEL F1 directly from the log
            for method, fold_results in result["method_f1"].items():
                problem_method_f1s.setdefault(method, {})
                problem_method_f1s[method].setdefault(key, []).extend(
                    x["f1"] for x in fold_results
                )

        # print("\nMacro F1 per problem (mean of fold-level F1)")
        for (configuration, domain, problem, neighborhood), f1s in sorted(problem_fold_f1s.items()):
            macro = sum(f1s) / len(f1s)
            # print(
            #     f"{configuration} | {domain} | {problem} | "
            #     f"neighborhood={neighborhood}: macro F1={macro:.4f} | folds={len(f1s)}"
            # )

        setup_macro = {}
        for (configuration, domain, problem, neighborhood), f1s in problem_fold_f1s.items():
            setup_macro.setdefault((configuration, neighborhood), []).append(
                sum(f1s) / len(f1s)
            )

        # print("\nMacro F1 by setup (mean of per-problem macro F1)")
        for (configuration, neighborhood), problem_means in sorted(
            setup_macro.items(),
            key=lambda item: sum(item[1]) / len(item[1]),
            reverse=True,
        ):
            macro = sum(problem_means) / len(problem_means)
            # print(
            #     f"{configuration} | neighborhood={neighborhood}: "
            #     f"macro F1={macro:.4f} | problems={len(problem_means)}"
            # )


    # problem_overview = {}
    # for result in results:
    #     key = (
    #         result["domain"],
    #         result["problem"],
    #         result["configuration"],
    #         result["neighborhood"],
    #     )
    #     summary = problem_overview.setdefault(
    #         key,
    #         {"runs": 0, "complete_runs": 0, "folds": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
    #     )
    #     summary["runs"] += 1
    #     summary["complete_runs"] += len(result["folds"]) == 10
    #     for fold in result["folds"]:
    #         summary["folds"] += 1
    #         for metric in ("tp", "fp", "tn", "fn"):
    #             summary[metric] += fold[metric]

    # problem_candidates = {}
    # for (domain, problem, configuration, neighborhood), summary in problem_overview.items():
    #     denominator = 2 * summary["tp"] + summary["fp"] + summary["fn"]
    #     f1 = 2 * summary["tp"] / denominator if denominator else 0.0
    #     problem_candidates.setdefault((domain, problem), []).append(
    #         (f1, configuration, neighborhood, summary)
    #     )

    # selected_winners = []
    # print("\nBest experiment and neighborhood for each problem")
    # for problem_key, candidates in sorted(problem_candidates.items()):
    #     complete_candidates = [
    #         candidate for candidate in candidates if candidate[3]["complete_runs"]
    #     ]
    #     ranked_candidates = complete_candidates or candidates
    #     f1, configuration, neighborhood, summary = max(
    #         ranked_candidates,
    #         key=lambda candidate: candidate[0],
    #     )
    #     selected_winners.append(summary)
    #     print(
    #         f"{problem_key[0]} | {problem_key[1]}: "
    #         f"best={configuration}, neighborhood={neighborhood}, "
    #         f"combined F1={f1:.4f} | "
    #         f"complete runs={summary['complete_runs']}/{summary['runs']} | "
    #         f"folds={summary['folds']} | "
    #         f"TP={summary['tp']}, FP={summary['fp']}, "
    #         f"TN={summary['tn']}, FN={summary['fn']}"
    #     )

    #     joint_totals = {
    #         metric: sum(summary[metric] for summary in selected_winners)
    #     for metric in ("tp", "fp", "tn", "fn")
    # }
    # denominator = 2 * joint_totals["tp"] + joint_totals["fp"] + joint_totals["fn"]
    # joint_f1 = 2 * joint_totals["tp"] / denominator if denominator else 0.0
    # print("\nJoint score across the selected best combination for each problem")
    # print(
    #     f"combined F1={joint_f1:.4f} | problems={len(selected_winners)} | "
    #     f"TP={joint_totals['tp']}, FP={joint_totals['fp']}, "
    #     f"TN={joint_totals['tn']}, FN={joint_totals['fn']}"
    # )
    # CELOE
        # print("\nF1 per problem")

        # for key in sorted(problem_fold_f1s):
        #     configuration, domain, problem, neighborhood = key

        #     ndlm_f1s = problem_fold_f1s[key]
        #     ndlm_macro = sum(ndlm_f1s) / len(ndlm_f1s)

        #     print(f"{domain} | {problem} | neighborhood={neighborhood}")
        #     print(f"  NDLM: {ndlm_macro:.4f}")

        #     for method in sorted(problem_method_f1s):
        #         f1s = problem_method_f1s[method].get(key, [])

        #         if f1s:
        #             macro = sum(f1s) / len(f1s)
        #             print(f"  {method}: {macro:.4f}")
        #         else:
        #             print(f"  {method}: N/A")
        print("\nOverall macro F1 ")

        all_methods = {
            "NDLM": problem_fold_f1s,
            **problem_method_f1s,
        }

        for method, data in all_methods.items():
            problem_means = [
                sum(f1s) / len(f1s)
                for f1s in data.values()
                if f1s
            ]

            macro_f1 = sum(problem_means) / len(problem_means)

            print(
                f"{method}: macro F1={macro_f1:.4f} | "
                f"problems={len(problem_means)}"
            )
        # celoe = result["method_f1"]["CELOE"]
        # if celoe:
        #     celoe_f1s = [x["f1"] for x in celoe]
        #     print(
        #         f"  CELOE F1: {sum(celoe_f1s) / len(celoe_f1s):.4f} "
        #         f"({len(celoe_f1s)} folds)"
        #     )
        #     print(
        #         "    "
        #         + "; ".join(
        #             f"F{x['fold']}: {x['f1']:.4f}"
        #             for x in celoe
        #         )
        #     )
        # else:
        #     print("  CELOE F1: not available")

    # # OCEL
    # ocel = result["method_f1"]["OCEL"]
    # if ocel:
    #     ocel_f1s = [x["f1"] for x in ocel]
    #     print(
    #         f"  OCEL F1: {sum(ocel_f1s) / len(ocel_f1s):.4f} "
    #         f"({len(ocel_f1s)} folds)"
    #     )
    #     print(
    #         "    "
    #         + "; ".join(
    #             f"F{x['fold']}: {x['f1']:.4f}"
    #             for x in ocel
    #         )
    #     )
    # else:
    #     print("  OCEL F1: not available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
            type=Path,
            default=Path("outputs/CONCEPT_LEARNER/BY_EXAMPLE_4"),
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--check-benchmarks", action="store_true", help="Check benchmark results.")
    args = parser.parse_args()
    for name in names:
        print("parsing domain:", name)
        results = collect_results(args.root, check_benchmarks=args.check_benchmarks, domain_name= name )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_text(results, check_benchmarks=args.check_benchmarks)


if __name__ == "__main__":
    main()
