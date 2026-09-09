#!/usr/bin/env python3
"""Generate approach-by-domain LaTeX F1 tables from BENCHMARKS_BEST outputs."""

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from analyze_owl_cv_results import collect_results, fold_f1


KNOWN_DOMAINS = ("Biopax", "Carcinogenesis_drill", "Family", "Mutagenesis_drill")
UNKNOWN_DOMAINS = ("Carcinogenesis", "Lymphography", "Mutagenesis", "Nctrer")
DOMAIN_LABELS = {
    "Biopax": "Biopax",
    "Carcinogenesis": "Carcinogenesis",
    "Carcinogenesis_drill": "Carcinogenesis*",
    "Family": "Family",
    "Lymphography": "Lymphography",
    "Mutagenesis": "Mutagenesis",
    "Mutagenesis_drill": "Mutagenesis*",
    "Nctrer": "Nctrer",
}
PARADIGMS = {
    "NDLM": "Neural",
    "ALCSAT": "SAT-based",
    "CELOE": "Search-based",
    "Drill": "Neuro-symbolic search",
    "EvoLearner": "Evolutionary search",
    "OCEL": "Search-based",
    "TDL": "Search-based",
}
METHOD_ORDER = ("NDLM", "ALCSAT", "CELOE", "Drill", "EvoLearner", "OCEL", "TDL")


def result_scores(results: list[dict]) -> dict[str, tuple[float, float, int]]:
    """Return each method's best mean F1, LP standard deviation, and LP count."""
    problem_scores = defaultdict(list)

    for result in results:
        setup = ("NDLM", result["configuration"], result["neighborhood"], result["problem"])
        problem_scores[setup].extend(fold_f1(fold) for fold in result["folds"])
        for method, folds in result["method_f1"].items():
            problem_scores[(method, *setup[1:])].extend(item["f1"] for item in folds)

    setup_scores = defaultdict(list)
    for (method, configuration, neighborhood, _), fold_scores in problem_scores.items():
        if fold_scores:
            setup_scores[(method, configuration, neighborhood)].append(
                sum(fold_scores) / len(fold_scores)
            )

    scores_by_method = defaultdict(list)
    for (method, _, _), problem_means in setup_scores.items():
        scores_by_method[method].append((
            sum(problem_means) / len(problem_means),
            statistics.stdev(problem_means) if len(problem_means) > 1 else 0.0,
            len(problem_means),
        ))
    return {
        method: max(scores, key=lambda score: score[0])
        for method, scores in scores_by_method.items()
        if scores
    }


def collect_domain_scores(root: Path, domain: str) -> dict[str, tuple[float, float, int]]:
    results = collect_results(
        root,
        check_benchmarks=True,
        domain_name=domain,
        common_ndlm_nlm_only=False,
    )
    return result_scores(results)


def latex_table(domains: tuple[str, ...], scores: dict[str, dict[str, tuple[float, float, int]]], title: str, label: str) -> str:
    headers = " & ".join(f"\\textbf{{{DOMAIN_LABELS[domain]}}}" for domain in domains)
    maxima = {
        domain: max(
            (scores[domain].get(method, (float("-inf"), 0.0, 0))[0] for method in METHOD_ORDER),
            default=float("-inf"),
        )
        for domain in domains
    }
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{ll|{'c' * len(domains)}}}",
        "\\toprule",
        f"\\textbf{{Approach}} & \\textbf{{Paradigm}} & \\multicolumn{{{len(domains)}}}{{c}}{{\\textbf{{{title}}}}} \\\\",
        f" & & {headers} \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        values = []
        for domain in domains:
            score_summary = scores[domain].get(method)
            if score_summary is None:
                values.append("--")
            else:
                score, standard_deviation, problem_count = score_summary
                rendered = f"{100 * score:.2f}"
                if problem_count > 1:
                    rendered += f" $\\pm$ {100 * standard_deviation:.2f}"
                if abs(score - maxima[domain]) < 1e-12:
                    rendered = f"\\textbf{{{rendered}}}"
                values.append(rendered)
        lines.append(f"{method} & {PARADIGMS[method]} & {' & '.join(values)} \\\\")
    lines.extend((
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Macro F1 scores reported as percentages. For domains with multiple learning problems, scores are reported as mean $\\pm$ sample standard deviation across learning problems. Each method uses its best available configuration and neighborhood; best per-domain means are in bold.}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/CONCEPT_LEARNER/BENCHMARKS_BEST"),
        help="Root directory containing BENCHMARKS_BEST result logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .tex file to write; prints to stdout when omitted.",
    )
    args = parser.parse_args()

    all_domains = KNOWN_DOMAINS + UNKNOWN_DOMAINS
    scores = {domain: collect_domain_scores(args.root, domain) for domain in all_domains}
    output = "\n\n".join((
        latex_table(KNOWN_DOMAINS, scores, "Known ground truth formula", "tab:approach_vs_known_domain"),
        latex_table(UNKNOWN_DOMAINS, scores, "Unknown formula", "tab:approach_vs_unknown_domain"),
    )) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote LaTeX tables to: {args.output}")


if __name__ == "__main__":
    main()