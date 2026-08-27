#!/usr/bin/env python3
"""Summarize success metrics from output.log files under eval_results run folders.

This script recursively finds directories named "eval_results" under a root path.
For each immediate subdirectory (treated as a run), it parses output.log and reports:
- Total Successes x / y (from "Total Successes" line when present)
- Train successes and totals from lines starting with "file: ..." with "/train/"
- Test successes and totals from lines starting with "file: ..." with "/test/"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
import sys

f = open("other/eval_results_success_summary_NLDM.log", "w")
sys.stdout = f
sys.stderr = f

TOTAL_SUCCESS_RE = re.compile(
    r"Total\s+Su(?:ccesses|cesses)\s*:\s*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
FILE_RESULT_RE = re.compile(
    r"file:\s*([^,]+),.*?success:\s*(True|False)",
    re.IGNORECASE,
)


@dataclass
class Counter:
    successes: int = 0
    total: int = 0

    def add(self, success: bool) -> None:
        self.total += 1
        if success:
            self.successes += 1

    def add_counter(self, other: "Counter") -> None:
        self.successes += other.successes
        self.total += other.total


@dataclass
class RunSummary:
    run_dir: str
    log_path: str
    total_successes_line: Optional[Counter]
    train: Counter
    test: Counter
    other: Counter


@dataclass
class ExperimentSummary:
    experiment_dir: str
    eval_results_dir: str
    runs_with_log: int = 0
    runs_missing_log: int = 0
    total_successes_line: Counter = field(default_factory=Counter)
    total_successes_rows: Counter = field(default_factory=Counter)
    train: Counter = field(default_factory=Counter)
    test: Counter = field(default_factory=Counter)
    other: Counter = field(default_factory=Counter)


def find_eval_results_dirs(root: str) -> List[str]:
    matches: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "eval_results":
            matches.append(dirpath)
            # Skip descending into this eval_results directory.
            dirnames[:] = []
            continue

        # Record eval_results children and prune them so os.walk never enters them.
        eval_children = [name for name in dirnames if name == "eval_results"]
        for name in eval_children:
            matches.append(os.path.join(dirpath, name))
            dirnames.remove(name)
    return sorted(matches)


def iter_run_dirs(eval_results_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(eval_results_dir)):
        path = os.path.join(eval_results_dir, name)
        if os.path.isdir(path):
            yield path


def parse_output_log(log_path: str) -> RunSummary:
    total_successes_line: Optional[Counter] = None
    train = Counter()
    test = Counter()
    other = Counter()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Fast pre-filters reduce costly regex work on large action traces.
            if "Total Su" in line or "Total su" in line:
                total_match = TOTAL_SUCCESS_RE.search(line)
                if total_match:
                    total_successes_line = Counter(
                        successes=int(total_match.group(1)),
                        total=int(total_match.group(2)),
                    )

            if "file:" not in line or "success:" not in line:
                continue

            file_match = FILE_RESULT_RE.search(line)
            if not file_match:
                continue

            file_path = file_match.group(1)
            success = file_match.group(2).lower() == "true"

            if "/train/" in file_path:
                train.add(success)
            elif "/test/" in file_path:
                test.add(success)
            else:
                other.add(success)

    return RunSummary(
        run_dir=os.path.basename(os.path.dirname(log_path)),
        log_path=log_path,
        total_successes_line=total_successes_line,
        train=train,
        test=test,
        other=other,
    )


def format_counter(counter: Counter) -> str:
    if counter.total == 0:
        return "0 / 0"
    return f"{counter.successes} / {counter.total} ({100.0 * counter.successes / counter.total:.2f}%)"


def parse_work_item(item: Tuple[str, str]) -> Tuple[str, RunSummary]:
    run_name, log_path = item
    return run_name, parse_output_log(log_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find eval_results directories and summarize output.log "
            "success metrics for each run subdirectory."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to search recursively (default: current directory)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also print runs where output.log is missing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 1) * 2)),
        help="Number of parallel workers for parsing output.log files",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    eval_results_dirs = find_eval_results_dirs(root)

    if not eval_results_dirs:
        print(f"No eval_results directories found under: {root}")
        return 0

    for eval_dir in eval_results_dirs:
        experiment_dir = os.path.dirname(eval_dir)
        summary = ExperimentSummary(
            experiment_dir=experiment_dir,
            eval_results_dir=eval_dir,
        )

        print(f"experiment: {experiment_dir}")
        print(f"  eval_results: {eval_dir}")
        run_dirs = list(iter_run_dirs(eval_dir))
        if not run_dirs:
            print("  (no run subdirectories)")
            print()
            continue

        parse_items: List[Tuple[str, str]] = []

        for run_dir in run_dirs:
            log_path = os.path.join(run_dir, "output.log")
            run_name = os.path.basename(run_dir)

            if not os.path.isfile(log_path):
                summary.runs_missing_log += 1
                if args.include_empty:
                    print(f"  run: {run_name}")
                    print("    output.log: missing")
                continue

            summary.runs_with_log += 1
            parse_items.append((run_name, log_path))

        parsed_runs: List[Tuple[str, RunSummary]] = []
        if parse_items:
            if args.workers <= 1:
                parsed_runs = [parse_work_item(item) for item in parse_items]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    parsed_runs = list(executor.map(parse_work_item, parse_items))

        for run_name, run_summary in parsed_runs:

            train_test_total = Counter(
                successes=run_summary.train.successes + run_summary.test.successes + run_summary.other.successes,
                total=run_summary.train.total + run_summary.test.total + run_summary.other.total,
            )

            if run_summary.total_successes_line is not None:
                summary.total_successes_line.add_counter(run_summary.total_successes_line)

            summary.total_successes_rows.add_counter(train_test_total)
            summary.train.add_counter(run_summary.train)
            summary.test.add_counter(run_summary.test)
            summary.other.add_counter(run_summary.other)

            print(f"  run: {run_name}")
            if run_summary.total_successes_line is None:
                print("    total successes (from line): not found")
            else:
                print(
                    "    total successes (from line): "
                    f"{run_summary.total_successes_line.successes} / {run_summary.total_successes_line.total}"
                )
            print(f"    total successes (from file rows): {format_counter(train_test_total)}")
            print(f"    train successes: {format_counter(run_summary.train)}")
            print(f"    test successes: {format_counter(run_summary.test)}")
            if run_summary.other.total > 0:
                print(f"    other successes: {format_counter(run_summary.other)}")

        print("  summary")
        print(f"    runs with output.log: {summary.runs_with_log}")
        print(f"    runs missing output.log: {summary.runs_missing_log}")
        if summary.total_successes_line.total > 0:
            print(
                "    total successes (sum of \"Total Successes\" lines): "
                f"{summary.total_successes_line.successes} / "
                f"{summary.total_successes_line.total} "
                f"({100.0 * summary.total_successes_line.successes / summary.total_successes_line.total:.2f}%)"
            )
        else:
            print('    total successes (sum of "Total Successes" lines): not found')

        print(
            "    total successes (from file rows): "
            f"{format_counter(summary.total_successes_rows)}"
        )
        print(f"    train successes: {format_counter(summary.train)}")
        print(f"    test successes: {format_counter(summary.test)}")
        if summary.other.total > 0:
            print(f"    other successes: {format_counter(summary.other)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
