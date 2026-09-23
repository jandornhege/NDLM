#!/usr/bin/env python3
"""Summarize ActionModel misclassification counts across domains and runs.

This script scans a root directory for ActionModel output.log files and extracts,
for each domain/run/action pair, the per-action misclassification counts reported by
lines of the form:

    Overview misclassifications for action pickup: train #c_miss=0, train #r_miss=0, test #c_miss=0, test #r_miss=0

It is designed for repeated runs in the same domain as well as multiple actions per
run (e.g. pickup, putdown, stack, unstack).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

OVERVIEW_RE = re.compile(
    r"Overview misclassifications for action (?P<action>[^:]+): "
    r"train #c_miss=(?P<train_c>\d+), train #r_miss=(?P<train_r>\d+), "
    r"test #c_miss=(?P<test_c>\d+), test #r_miss=(?P<test_r>\d+)"
)

RUN_ID_HINTS = {"blocks","ActionModel", "RERUN", "RERUN_shard", "L1", "L2", "L3", "c0", "c1", "c2", "c3", "c4", "c5"}
CONFIG_DIR_HINTS = {"eff", "prec", "sandbox", "test_only", "ActionModel", "optGP"}


def is_run_like_name(name: str) -> bool:
    value = str(name).strip()
    if not value:
        return False
    if value.startswith("run_") or value.startswith("output_") or value.startswith("RERUN"):
        return True
    if value in RUN_ID_HINTS:
        return True
    if re.fullmatch(r"c\d+", value):
        return False
    if value.isdigit():
        return True
    if re.fullmatch(r"\d+_\d+", value):
        return True
    return False


def normalize_run_name(run_name: str) -> str:
    run_name = str(run_name).strip()
    if not run_name:
        return run_name
    if run_name.startswith("run_"):
        return run_name
    return f"run_{run_name}"


def infer_domain_and_run(log_path: Path, root: Path) -> tuple[str, str]:
    rel_parts = log_path.relative_to(root).parts
    if not rel_parts:
        return "unknown", log_path.parent.name

    for index, part in enumerate(rel_parts):
        if part == "sandbox" and index + 2 < len(rel_parts):
            return rel_parts[index + 1], rel_parts[index + 2]
        if part == "test_only" and index + 2 < len(rel_parts):
            return rel_parts[index + 1], rel_parts[index + 2]

    ancestors = list(rel_parts[:-1])
    if not ancestors:
        return "unknown", log_path.parent.name

    for index in range(len(ancestors) - 1, -1, -1):
        part = ancestors[index]
        if part in CONFIG_DIR_HINTS or re.fullmatch(r"c\d+", str(part)):
            continue
        if is_run_like_name(part):
            run_name = part
            for candidate_index in range(index - 1, -1, -1):
                candidate = ancestors[candidate_index]
                if candidate in CONFIG_DIR_HINTS or re.fullmatch(r"c\d+", str(candidate)):
                    continue
                if not is_run_like_name(candidate):
                    return candidate, run_name
            for candidate_index in range(index + 1, len(ancestors)):
                candidate = ancestors[candidate_index]
                if candidate in CONFIG_DIR_HINTS or re.fullmatch(r"c\d+", str(candidate)):
                    continue
                if not is_run_like_name(candidate):
                    return candidate, run_name
            break

    # Legacy layout: root/<domain>/<timestamp>/output.log
    if len(ancestors) >= 2:
        domain_name = ancestors[-2]
        run_name = ancestors[-1]
        if is_run_like_name(run_name) and not is_run_like_name(domain_name):
            return domain_name, run_name

    for name in reversed(ancestors):
        if is_run_like_name(name):
            continue
        if name in CONFIG_DIR_HINTS or re.fullmatch(r"c\d+", str(name)):
            continue
        return name, log_path.parent.name

    return ancestors[-1], log_path.parent.name


def find_matching_sandbox_log(root: Path, domain: str, run: str) -> Path | None:
    run_name = normalize_run_name(run)
    sandbox_root = root / "sandbox"
    if sandbox_root.exists():
        for candidate in [
            sandbox_root / domain / run_name / "output.log",
            sandbox_root / domain / run / "output.log",
            sandbox_root / domain / run_name / "output_*.log",
            sandbox_root / domain / run / "output_*.log",
        ]:
            if candidate.exists() and candidate.is_file():
                return candidate

    for candidate in sorted(root.rglob("output.log")):
        if "sandbox" not in candidate.relative_to(root).parts:
            continue
        c_domain, c_run = infer_domain_and_run(candidate, root)
        if c_domain == domain and normalize_run_name(c_run) == run_name:
            return candidate
    return None


def parse_log(log_path: Path) -> list[dict[str, int | str]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, int | str]] = []
    for match in OVERVIEW_RE.finditer(text):
        action = match.group("action").strip()
        records.append(
            {
                "action": action,
                "train_c": int(match.group("train_c")),
                "train_r": int(match.group("train_r")),
                "test_c": int(match.group("test_c")),
                "test_r": int(match.group("test_r")),
            }
        )
    return records


def iter_action_model_logs(root: Path):
    for log_path in sorted(root.rglob("output.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Starting task: ActionModel" in text or "Overview misclassifications for action" in text:
            yield log_path


def summarize_root(root: Path, *, base_root: Path | None = None, domain_run_action=None, domain_totals=None, run_totals=None, perfect_runs=None, train_perfect_runs=None):
    if base_root is None:
        base_root = root
    seen_any = False
    for log_path in iter_action_model_logs(root):
        records = parse_log(log_path)
        if not records:
            continue
        seen_any = True

        domain, run = infer_domain_and_run(log_path, base_root)
        sandbox_log = find_matching_sandbox_log(base_root, domain, run)
        if sandbox_log is not None and sandbox_log != log_path:
            sandbox_records = parse_log(sandbox_log)
            if sandbox_records:
                records = sandbox_records

        run_total = 0
        run_perfect = True
        run_train_perfect = True
        for record in records:
            action = str(record["action"])
            train_total = int(record["train_c"]) + int(record["train_r"])
            test_total = int(record["test_c"]) + int(record["test_r"])
            total = train_total + test_total
            run_total += total
            if total > 0:
                run_perfect = False
            if train_total > 0:
                run_train_perfect = False
            domain_run_action[(domain, run, action)] = {
                "train_c": int(record["train_c"]),
                "train_r": int(record["train_r"]),
                "test_c": int(record["test_c"]),
                "test_r": int(record["test_r"]),
                "train_total": train_total,
                "test_total": test_total,
                "total": total,
            }
            domain_totals[domain] += total
            run_totals[(domain, run)] += total
        perfect_runs[domain] += int(run_perfect)
        train_perfect_runs[domain] += int(run_train_perfect)

    return seen_any


def iter_roots(root: Path, iterate: bool):
    if not iterate:
        yield root
        return

    candidate_dirs: set[Path] = set()
    for log_path in sorted(root.rglob("output.log")):
        run_dir = log_path.parent
        rel_parts = run_dir.relative_to(root).parts
        if len(rel_parts) >= 4:
            candidate_dirs.add(run_dir.parents[1])
        elif len(rel_parts) >= 2:
            candidate_dirs.add(run_dir.parent)
        else:
            candidate_dirs.add(root)

    if not candidate_dirs:
        yield root
        return

    for candidate in sorted(candidate_dirs):
        yield candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs/NDLM_OPTGP_final_0_shard",
        help="Root directory to scan for ActionModel output.log files.",
    )
    parser.add_argument(
        "--by-domain",
        action="store_true",
        help="Print aggregated totals per domain instead of per run/action rows.",
    )
    parser.add_argument(
        "--iterate",
        "--itterate",
        dest="iterate",
        action="store_true",
        help="Process each run directory containing an output.log under the root and aggregate the results across them.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    if args.iterate:
        for candidate_root in iter_roots(root, True):
            local_domain_run_action = defaultdict(list)
            local_domain_totals = defaultdict(int)
            local_run_totals = defaultdict(int)
            local_perfect_runs = defaultdict(int)
            local_train_perfect_runs = defaultdict(int)
            seen_any = summarize_root(
                candidate_root,
                base_root=root,
                domain_run_action=local_domain_run_action,
                domain_totals=local_domain_totals,
                run_totals=local_run_totals,
                perfect_runs=local_perfect_runs,
                train_perfect_runs=local_train_perfect_runs,
            )
            if not seen_any:
                continue

            header = f"Summary for {candidate_root.relative_to(root) if candidate_root != root else root.name}"
            print(f"\n{header}")
            print(f"{'domain':<28} {'runs':>6} {'perfect_runs':>12} {'actions':>8} {'misclassifications':>18}")
            print("-" * 90)
            for domain in sorted({d for d, _, _ in local_domain_run_action.keys()}):
                actions = {action for d, _, action in local_domain_run_action if d == domain}
                runs = {run for d, run, _ in local_domain_run_action if d == domain}
                total = sum(v["total"] for (d, _, _), v in ((k, v) for k, v in local_domain_run_action.items() if k[0] == domain))
                print(f"{domain:<28} {len(runs):>6} {local_perfect_runs.get(domain, 0):>12} {len(actions):>8} {total:>18}")
        return 0

    domain_run_action = defaultdict(list)
    domain_totals = defaultdict(int)
    run_totals = defaultdict(int)
    perfect_runs = defaultdict(int)
    train_perfect_runs = defaultdict(int)
    seen_any = False

    for candidate_root in iter_roots(root, False):
        candidate_seen = summarize_root(
            candidate_root,
            base_root=root,
            domain_run_action=domain_run_action,
            domain_totals=domain_totals,
            run_totals=run_totals,
            perfect_runs=perfect_runs,
            train_perfect_runs=train_perfect_runs,
        )
        seen_any = seen_any or candidate_seen

    if not seen_any:
        raise SystemExit(
            f"No ActionModel output.log files with 'Starting task: ActionModel' or 'Overview misclassifications for action' were found under: {root}. "
            "This directory appears to be an optGP shard rather than an ActionModel output tree."
        )

    if args.by_domain:
        print(f"{'domain':<28} {'runs':>6} {'perfect_runs':>12} {'actions':>8} {'misclassifications':>18}")
        print("-" * 90)
        for domain in sorted({d for d, _, _ in domain_run_action.keys()}):
            actions = {action for d, _, action in domain_run_action if d == domain}
            runs = {run for d, run, _ in domain_run_action if d == domain}
            total = sum(v["total"] for (d, _, _), v in ((k, v) for k, v in domain_run_action.items() if k[0] == domain))
            print(f"{domain:<28} {len(runs):>6} {perfect_runs.get(domain, 0):>12} {len(actions):>8} {total:>18}")
        return 0

    print(f"{'domain':<22} {'run':<22} {'action':<18} {'train_c':>8} {'train_r':>8} {'test_c':>8} {'test_r':>8} {'total':>8}")
    print("-" * 110)
    for (domain, run, action), values in sorted(domain_run_action.items()):
        print(
            f"{domain:<22} {run:<22} {action:<18} "
            f"{values['train_c']:>8} {values['train_r']:>8} {values['test_c']:>8} {values['test_r']:>8} {values['total']:>8}"
        )

    print("\nAction summary: runs vs perfect runs")
    print(f"{'domain':<22} {'action':<18} {'runs':>6} {'perfect_runs':>12} {'train_zero_runs':>15} {'test_zero_runs':>15} {'both_zero_runs':>15}")
    print("-" * 120)
    action_counts = defaultdict(lambda: {"runs": 0, "perfect_runs": 0, "train_zero_runs": 0, "test_zero_runs": 0, "both_zero_runs": 0})
    for (domain, run, action), values in sorted(domain_run_action.items()):
        counts = action_counts[(domain, action)]
        counts["runs"] += 1
        if values["total"] == 0:
            counts["perfect_runs"] += 1

        train_zero = values["train_c"] == 0 and values["train_r"] == 0
        test_zero = values["test_c"] == 0 and values["test_r"] == 0
        if train_zero:
            counts["train_zero_runs"] += 1
        if test_zero:
            counts["test_zero_runs"] += 1
        if train_zero and test_zero:
            counts["both_zero_runs"] += 1

    for (domain, action) in sorted(action_counts):
        counts = action_counts[(domain, action)]
        print(f"{domain:<22} {action:<18} {counts['runs']:>6} {counts['perfect_runs']:>12} {counts['train_zero_runs']:>15} {counts['test_zero_runs']:>15} {counts['both_zero_runs']:>15}")

    print("\nPerfect runs by domain:")
    print(f"{'domain':<22} {'perfect_runs':>12} {'train_perfect_runs':>19} {'total_runs':>10}")
    print("-" * 80)
    for domain in sorted({d for d, _, _ in domain_run_action.keys()}):
        runs = {run for d_name, run, _ in domain_run_action if d_name == domain}
        perfect = perfect_runs.get(domain, 0)
        train_perfect = train_perfect_runs.get(domain, 0)
        print(f"{domain:<22} {perfect:>12} {train_perfect:>19} {len(runs):>10}")

    print("\nAggregated by domain:")
    print(f"{'domain':<22} {'runs':>6} {'actions':>8} {'misclassifications':>18}")
    print("-" * 70)
    for domain in sorted(domain_totals):
        runs = {run for d, run, _ in domain_run_action if d == domain}
        actions = {action for d, _, action in domain_run_action if d == domain}
        print(f"{domain:<22} {len(runs):>6} {len(actions):>8} {domain_totals[domain]:>18}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # This can happen when piping output to head/tail. Exit cleanly.
        pass
