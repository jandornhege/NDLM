#!/usr/bin/env python3
"""Print train/test evaluation success summaries for every test_only run."""

import argparse
import re
from pathlib import Path

INSTANCE_RE = re.compile(
    r"file:\s+(?P<path>.+?/(?P<split>train|test)/.*?\.pddl),\s+steps_taken:\s+\d+,\s+success:\s*(?P<success>True|False)"
)
TOTAL_RE = re.compile(r"Total Successes:\s*(\d+)\s*/\s*(\d+)\s*\(([^)]+)%\)")
OVERVIEW_RE = re.compile(
    r"Overview misclassifications for action (?P<action>[^:]+): "
    r"train #c_miss=(?P<train_c>\d+), train #r_miss=(?P<train_r>\d+), "
    r"test #c_miss=(?P<test_c>\d+), test #r_miss=(?P<test_r>\d+)"
)


def find_matching_sandbox_log(root: Path, domain: str, run_group: str) -> Path | None:
    sandbox_root = root / "sandbox" / domain / run_group
    if sandbox_root.is_dir():
        matches = sorted(sandbox_root.rglob("output.log"))
        if matches:
            return matches[0]

    for candidate in sorted((root / "sandbox").rglob("output.log")):
        rel = candidate.relative_to(root)
        parts = rel.parts
        if len(parts) >= 4 and parts[0] == "sandbox":
            if parts[1] == domain and parts[2] == run_group:
                return candidate
    return None


def summarize_log(log: Path) -> dict[str, tuple[int, int]]:
    counts = {"train": {"success": 0, "total": 0}, "test": {"success": 0, "total": 0}}
    for line in log.read_text(errors="replace").splitlines():
        match = INSTANCE_RE.search(line)
        if not match:
            continue
        split = match.group("split")
        counts[split]["total"] += 1
        if match.group("success") == "True":
            counts[split]["success"] += 1
    return {
        split: (stats["success"], stats["total"])
        for split, stats in counts.items()
    }


def format_summary(split: str, success: int, total: int) -> str:
    if total == 0:
        return f"eval {split} successes: 0 / 0 (0.00%)"
    rate = 100.0 * success / total
    return f"eval {split} successes: {success} / {total} ({rate:.2f}%)"


def format_instance_summary(split_counts: dict[str, tuple[int, int]]) -> str:
    train_total = split_counts["train"][1]
    test_total = split_counts["test"][1]
    return f"train instances={train_total}, test instances={test_total}"


def summarize_domain(domain_path: Path, experiment: Path) -> dict[str, float | int]:
    runs = []
    for log in sorted(domain_path.rglob("output.log")):
        text = log.read_text(errors="replace")
        counts = {"train": {"success": 0, "total": 0}, "test": {"success": 0, "total": 0}}
        for line in text.splitlines():
            match = INSTANCE_RE.search(line)
            if not match:
                continue
            split = match.group("split")
            counts[split]["total"] += 1
            if match.group("success") == "True":
                counts[split]["success"] += 1

        train_total = counts["train"]["total"]
        test_total = counts["test"]["total"]
        train_perfect = train_total > 0 and counts["train"]["success"] == train_total
        test_perfect = test_total > 0 and counts["test"]["success"] == test_total

        solved_count = 0
        total_count = 0
        for line in text.splitlines():
            match = TOTAL_RE.search(line)
            if match:
                solved_count = int(match.group(1))
                total_count = int(match.group(2))
                break

        runs.append(
            {
                "train_perfect": train_perfect,
                "test_perfect": test_perfect,
                "solved_count": solved_count,
                "total_count": total_count,
                "train_total": train_total,
                "test_total": test_total,
                "train_success": counts["train"]["success"],
                "test_success": counts["test"]["success"],
            }
        )

    total_runs = len(runs)
    if total_runs == 0:
        return {
            "train_runs": 0,
            "total_runs": 0,
            "test_runs": 0,
            "mean_count": 0.0,
            "mean_total": 0.0,
            "avg_train_instances": 0.0,
            "avg_test_instances": 0.0,
            "avg_train_success": 0.0,
            "avg_test_success": 0.0,
            "avg_train_total": 0.0,
            "avg_test_total": 0.0,
            "std_train_success": 0.0,
            "std_test_success": 0.0,
            "std_train_total": 0.0,
            "std_test_total": 0.0,
        }

    train_runs = sum(1 for run in runs if run["train_perfect"])
    test_runs = sum(1 for run in runs if run["test_perfect"])
    total_solved = sum(run["solved_count"] for run in runs)
    total_instances = sum(run["total_count"] for run in runs)
    train_success_vals = [run["train_success"] for run in runs]
    test_success_vals = [run["test_success"] for run in runs]
    train_total_vals = [run["train_total"] for run in runs]
    test_total_vals = [run["test_total"] for run in runs]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs):
        if len(xs) <= 1:
            return 0.0
        mu = mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    avg_train_success = mean(train_success_vals)
    avg_test_success = mean(test_success_vals)
    avg_train_total = mean(train_total_vals)
    avg_test_total = mean(test_total_vals)
    std_train_success = std(train_success_vals)
    std_test_success = std(test_success_vals)
    std_train_total = std(train_total_vals)
    std_test_total = std(test_total_vals)
    mean_count = total_solved / total_runs if total_runs > 0 else 0.0
    mean_total = total_instances / total_runs if total_runs > 0 else 0.0
    return {
        "train_runs": train_runs,
        "total_runs": total_runs,
        "test_runs": test_runs,
        "mean_count": mean_count,
        "mean_total": mean_total,
        "avg_train_instances": avg_train_success,
        "avg_test_instances": avg_test_success,
        "avg_train_success": avg_train_success,
        "avg_test_success": avg_test_success,
        "avg_train_total": avg_train_total,
        "avg_test_total": avg_test_total,
        "std_train_success": std_train_success,
        "std_test_success": std_test_success,
        "std_train_total": std_train_total,
        "std_test_total": std_test_total,
    }


def format_domain_summary(domain_name: str, stats: dict[str, float | int]) -> str:
    total_runs = int(stats["total_runs"])
    train_runs = int(stats["train_runs"])
    test_runs = int(stats["test_runs"])
    if total_runs == 0:
        return (
            f"{domain_name} & 0/0 (0.00%) & 0/0 (0.00%) & "
            "avg train solved 0.0 / total 0.0 & avg test solved 0.0 / total 0.0 & mean 0.0 / 0.0 (0.00%) \\\\"
        )

    train_pct = 100.0 * train_runs / total_runs
    test_pct = 100.0 * test_runs / total_runs
    avg_train_success = float(stats["avg_train_success"])
    avg_test_success = float(stats["avg_test_success"])
    avg_train_total = float(stats["avg_train_total"])
    avg_test_total = float(stats["avg_test_total"])
    std_train_success = float(stats["std_train_success"])
    std_test_success = float(stats["std_test_success"])
    std_train_total = float(stats["std_train_total"])
    std_test_total = float(stats["std_test_total"])
    mean_count = float(stats["mean_count"])
    mean_total = float(stats["mean_total"])
    mean_pct = 100.0 * (mean_count / mean_total) if mean_total > 0 else 0.0
    return (
        f"{domain_name} & {train_runs}/{total_runs} ({train_pct:.2f}%) & "
        f"{test_runs}/{total_runs} ({test_pct:.2f}%) & "
        f"avg train solved {avg_train_success:.1f} ± {std_train_success:.1f} / total {avg_train_total:.1f} ± {std_train_total:.1f} & "
        f"avg test solved {avg_test_success:.1f} ± {std_test_success:.1f} / total {avg_test_total:.1f} ± {std_test_total:.1f} & "
        f"mean {mean_count:.1f} / {mean_total:.1f} ({mean_pct:.2f}%) \\\\"
    )


def summarize_sandbox_misclassifications(log: Path | None) -> str:
    if log is None or not log.exists():
        return "sandbox train missclassifications: unavailable"

    train_total = 0
    action_count = 0
    nonzero_actions = 0
    for match in OVERVIEW_RE.finditer(log.read_text(errors="replace")):
        action_count += 1
        train_miss = int(match.group("train_c")) + int(match.group("train_r"))
        train_total += train_miss
        if train_miss > 0:
            nonzero_actions += 1

    if action_count == 0:
        return "sandbox train missclassifications: no overview lines found"
    return (
        f"sandbox train missclassifications: {train_total} total over {action_count} actions "
        f"({nonzero_actions} nonzero actions)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default="outputs/final_1__large_NDLM_OPTGP",
        help="Path to the experiment directory containing test_only/",
    )
    args = parser.parse_args()

    experiment = Path(args.experiment)
    test_only = experiment / "test_only"
    if not test_only.is_dir():
        raise SystemExit(f"No such directory: {test_only}")

    domain_names = {
        "blocks": "Blocksworld",
        "gripper": "Gripper",
        "logistics": "Logistics",
        "miconic": "Miconic",
        "navigation-xy": "Navigation-xy",
    }

    domain_summary_lines = []
    for domain in sorted(p for p in test_only.iterdir() if p.is_dir()):
        print(f"== {domain.name} ==")
        for log in sorted(domain.rglob("output.log")):
            run = log.parent.name
            run_group = log.parent.parent.name
            split_counts = summarize_log(log)
            total_matches = [
                match.group(0)
                for line in log.read_text(errors="replace").splitlines()
                if (match := TOTAL_RE.search(line))
            ]
            sandbox_log = find_matching_sandbox_log(experiment, domain.name, run_group)
            sandbox_summary = summarize_sandbox_misclassifications(sandbox_log)

            summaries = [
                format_summary(split, *split_counts[split])
                for split in ("train", "test")
                if split_counts[split][1] > 0
            ]
            instance_summary = format_instance_summary(split_counts)
            if summaries:
                summary = " | ".join(summaries)
                if total_matches:
                    summary = f"{summary} | {total_matches[-1]}"
                print(f"  {run}: {summary} | {instance_summary} | {sandbox_summary}")
            else:
                print(f"  {run}: no eval results | {instance_summary} | {sandbox_summary}")

        domain_stats = summarize_domain(domain, experiment)
        pretty_name = domain_names.get(domain.name, domain.name.replace("-", " ").title())
        summary_line = f"{format_domain_summary(pretty_name, domain_stats)}"
        domain_summary_lines.append(summary_line)
        print(summary_line)
        print()

    print("== domain summaries ==")
    for line in domain_summary_lines:
        print(line)


if __name__ == "__main__":
    main()
