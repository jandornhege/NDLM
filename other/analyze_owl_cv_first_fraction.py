#!/usr/bin/env python3
"""Analyze CV performance on the first fraction of learning problems.

Learning problems are ordered by their directory names. The same selected
problem names are used for every configuration within a domain, so missing or
partial configurations cannot change which problems count as the first 80%.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_DOMAINS = (
	"Biopax",
	"Carcinogenesis",
	"Carcinogenesis_drill",
	"Family",
	"Lymphography",
	"Mutagenesis",
	"Mutagenesis_drill",
	"Nctrer",
	"Suramin",
)

FOLD_PATTERN = re.compile(
	r"(?:Outer\s+)?Fold\s+(?P<fold>\d+):\s*"
	r"TP=(?P<tp>\d+),\s*"
	r"FP=(?P<fp>\d+),\s*"
	r"TN=(?P<tn>\d+),\s*"
	r"FN=(?P<fn>\d+)"
)


def parse_summary(summary_path: Path, root: Path) -> dict | None:
	"""Parse one summary.log under configuration/domain/.../run_.../summary.log."""
	try:
		relative_parts = summary_path.relative_to(root).parts
	except ValueError:
		return None

	if len(relative_parts) != 6:
		return None

	configuration, domain, neighborhood_name, problem, run_name, filename = relative_parts
	if filename != "summary.log":
		return None
	if not neighborhood_name.startswith("neighborhood_"):
		return None
	if not run_name.startswith("run_"):
		return None

	try:
		neighborhood = int(neighborhood_name.removeprefix("neighborhood_"))
		run_id = int(run_name.removeprefix("run_"))
	except ValueError:
		return None

	text = summary_path.read_text(encoding="utf-8", errors="replace")
	folds = {}
	for match in FOLD_PATTERN.finditer(text):
		folds[int(match["fold"])] = {
			"tp": int(match["tp"]),
			"fp": int(match["fp"]),
			"tn": int(match["tn"]),
			"fn": int(match["fn"]),
		}

	if not folds:
		return None

	return {
		"configuration": configuration,
		"domain": domain,
		"neighborhood": neighborhood,
		"problem": problem,
		"run": run_id,
		"folds": [folds[index] for index in sorted(folds)],
		"summary": str(summary_path),
	}


def keep_best_run_per_problem(results: list[dict]) -> list[dict]:
	"""Keep the most complete run for each config/domain/neighborhood/problem."""
	grouped = {}
	for result in results:
		key = (
			result["configuration"],
			result["domain"],
			result["neighborhood"],
			result["problem"],
		)
		current = grouped.get(key)
		rank = (len(result["folds"]), result["run"])
		if current is None or rank > (len(current["folds"]), current["run"]):
			grouped[key] = result
	return list(grouped.values())


def collect_results(root: Path, domain: str) -> list[dict]:
	results = []
	for summary_path in root.glob(f"*/{domain}/**/summary.log"):
		result = parse_summary(summary_path, root)
		if result is not None:
			results.append(result)
	return keep_best_run_per_problem(results)


def select_first_fraction(results: list[dict], fraction: float) -> tuple[list[str], int]:
	"""Return the canonical problem names and requested count for one domain."""
	problem_names = sorted({result["problem"] for result in results})
	requested_count = max(1, int(len(problem_names) * fraction))
	return problem_names[:requested_count], requested_count


def confusion_totals(folds: list[dict]) -> dict[str, int]:
	return {
		metric: sum(fold[metric] for fold in folds)
		for metric in ("tp", "fp", "tn", "fn")
	}


def f1_from_counts(counts: dict[str, int]) -> float:
	denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
	return 2 * counts["tp"] / denominator if denominator else 0.0


def summarize_domain(
	results: list[dict],
	fraction: float,
	include_incomplete: bool,
) -> dict:
	selected_problems, requested_count = select_first_fraction(results, fraction)
	selected_set = set(selected_problems)
	grouped = defaultdict(list)
	for result in results:
		if result["problem"] in selected_set:
			grouped[(result["configuration"], result["neighborhood"])].append(result)

	configurations = []
	for (configuration, neighborhood), problem_results in grouped.items():
		all_folds = [fold for result in problem_results for fold in result["folds"]]
		totals = confusion_totals(all_folds)
		per_problem_f1 = [
			f1_from_counts(confusion_totals(result["folds"]))
			for result in problem_results
		]
		configurations.append(
			{
				"configuration": configuration,
				"neighborhood": neighborhood,
				"selected_problems": len(problem_results),
				"requested_problems": requested_count,
				"folds": len(all_folds),
				"complete": len(problem_results) == requested_count,
				"micro_f1": f1_from_counts(totals),
				"macro_f1": sum(per_problem_f1) / len(per_problem_f1),
				**totals,
			}
		)

	if not include_incomplete:
		configurations = [
			row for row in configurations
			if row["complete"]
		]
	configurations.sort(key=lambda row: row["macro_f1"], reverse=True)
	return {
		"domain": results[0]["domain"] if results else None,
		"total_problems": len({result["problem"] for result in results}),
		"selected_problems": selected_problems,
		"fraction": fraction,
		"include_incomplete": include_incomplete,
		"configurations": configurations,
	}


def print_text(summary: dict) -> None:
	domain = summary["domain"]
	print(f"\n{domain}")
	print("=" * len(domain))
	print(
		f"Using {len(summary['selected_problems'])}/"
		f"{summary['total_problems']} learning problems "
		f"({summary['fraction']:.1%}; sorted by problem name)."
	)
	for rank, row in enumerate(summary["configurations"], start=1):
		print(
			f"{rank}. {row['configuration']} | "
			f"neighborhood={row['neighborhood']} | "
			f"macro F1={row['macro_f1']:.4f} | "
			f"micro F1={row['micro_f1']:.4f} | "
			f"problems={row['selected_problems']}/{row['requested_problems']} | "
			f"folds={row['folds']} | "
			f"complete={row['complete']}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--root",
		type=Path,
		default=Path("outputs/CONCEPT_LEARNER/ALL"),
		help="Root directory containing configuration/domain output directories.",
	)
	parser.add_argument(
		"--fraction",
		type=float,
		default=0.8,
		help="Fraction of sorted learning problems to include; default: 0.8.",
	)
	parser.add_argument(
		"--domain",
		action="append",
		dest="domains",
		help="Domain to analyze; may be supplied more than once. Defaults to all domains.",
	)
	parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
	parser.add_argument(
		"--include-incomplete",
		action="store_true",
		help="Include configurations missing one or more selected learning problems.",
	)
	args = parser.parse_args()

	if not 0.0 < args.fraction <= 1.0:
		parser.error("--fraction must be greater than 0 and at most 1")
	if not args.root.is_dir():
		parser.error(f"output root does not exist: {args.root}")

	domains = args.domains or DEFAULT_DOMAINS
	summaries = []
	for domain in domains:
		results = collect_results(args.root, domain)
		if not results:
			print(f"No complete summaries found for {domain}.")
			continue
		summary = summarize_domain(
			results,
			args.fraction,
			include_incomplete=args.include_incomplete,
		)
		summaries.append(summary)
		if not args.json:
			print_text(summary)

	if args.json:
		print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
	main()
