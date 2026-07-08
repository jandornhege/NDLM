#!/usr/bin/env python3

"""Summarize NDLM/NLM run logs under an outputs directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


TRAIN_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\(Dataset\s+(\d+)\)\s+\|\s+Average Accuracy:\s+\(([^,]+),\s*([^\)]+)\)"
)
TEST_RE = re.compile(
    r"^Test Accuracy\s+\(Dataset\s+(\d+)\):\s+c:([^,]+),\s*r:(.+)$"
)
TRAIN_MISS_RE = re.compile(
    r"^\|\s*Total Misclassifications:\s*\(([-+]?\d+),\s*([-+]?\d+)\)"
)
TEST_MISS_RE = re.compile(
    r"^Test Misclassifications\s+\(Dataset\s+(\d+)\):\s+c:([-+]?\d+),\s*r:([-+]?\d+)"
)
LOSS_RE = re.compile(r"^\s*\|\s*Loss:\s+(.+)$")
LOADED_RE = re.compile(r"^Loaded\s+(\d+)\s+training samples and\s+(\d+)\s+testing samples\.$")
ACTION_RE = re.compile(r"^Action:\s*([^,]+),")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NDLM outputs and print an overview of run metrics."
    )
    parser.add_argument(
        "outputs_dir",
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "outputs",
        type=Path,
        help="Path to outputs directory (default: NDLM/outputs).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write parsed run summaries as JSON.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include runs without test metrics in aggregate stats.",
    )
    return parser.parse_args()


def to_float(value: str) -> float | None:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"nan", "none"}:
        return math.nan if lowered == "nan" else None
    try:
        return float(value)
    except ValueError:
        return None


def float_to_str(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def parse_log_file(log_path: Path, outputs_dir: Path) -> dict:
    rel_parts = log_path.relative_to(outputs_dir).parts
    model_bucket = rel_parts[0] if len(rel_parts) > 0 else "unknown"
    c_setting = rel_parts[1] if len(rel_parts) > 1 else "unknown"
    task_name = rel_parts[2] if len(rel_parts) > 2 else "unknown"

    def new_segment() -> dict:
        return {
            "latest_epoch": None,
            "train_c_values": [],
            "train_r_values": [],
            "test_c_values": [],
            "test_r_values": [],
            "train_miss_pairs": [],
            "test_miss_pairs": [],
            "saw_nan_loss": False,
        }

    def finalize_segment(segment: dict) -> dict:
        valid_train_c = [v for v in segment["train_c_values"] if isinstance(v, float) and not math.isnan(v)]
        valid_train_r = [v for v in segment["train_r_values"] if isinstance(v, float) and not math.isnan(v)]
        valid_test_c = [v for v in segment["test_c_values"] if isinstance(v, float) and not math.isnan(v)]
        valid_test_r = [v for v in segment["test_r_values"] if isinstance(v, float) and not math.isnan(v)]
        train_last_miss = segment["train_miss_pairs"][-1] if segment["train_miss_pairs"] else None
        test_last_miss = segment["test_miss_pairs"][-1] if segment["test_miss_pairs"] else None
        return {
            "epochs_seen": segment["latest_epoch"],
            "train_last_c": valid_train_c[-1] if valid_train_c else None,
            "train_best_c": max(valid_train_c) if valid_train_c else None,
            "train_last_r": valid_train_r[-1] if valid_train_r else None,
            "train_best_r": max(valid_train_r) if valid_train_r else None,
            "test_last_c": valid_test_c[-1] if valid_test_c else None,
            "test_best_c": max(valid_test_c) if valid_test_c else None,
            "test_last_r": valid_test_r[-1] if valid_test_r else None,
            "test_best_r": max(valid_test_r) if valid_test_r else None,
            "test_evals": len(valid_test_c),
            "train_last_miss_c": train_last_miss[0] if train_last_miss is not None else None,
            "train_last_miss_r": train_last_miss[1] if train_last_miss is not None else None,
            "test_last_miss_c": test_last_miss[0] if test_last_miss is not None else None,
            "test_last_miss_r": test_last_miss[1] if test_last_miss is not None else None,
            "last_train_success_zero_miss": bool(train_last_miss == (0, 0)) if train_last_miss is not None else False,
            "last_test_success_zero_miss": bool(test_last_miss == (0, 0)) if test_last_miss is not None else False,
            "nan_loss": segment["saw_nan_loss"],
        }

    def mean_or_none(values: list[float | None]) -> float | None:
        valid = [v for v in values if v is not None and not math.isnan(v)]
        if not valid:
            return None
        return sum(valid) / len(valid)

    segments: list[dict] = []
    current = new_segment()

    with log_path.open("r", encoding="utf-8") as handle:
        lines = [raw_line.strip() for raw_line in handle]

    action_names = []
    seen_actions = set()
    for line in lines:
        action_match = ACTION_RE.match(line)
        if action_match:
            action_name = action_match.group(1).strip()
            if action_name not in seen_actions:
                action_names.append(action_name)
                seen_actions.add(action_name)

    for line in lines:

            loaded_match = LOADED_RE.match(line)
            if loaded_match:
                # Start a new subproblem run when a new train/test block begins.
                if (
                    current["latest_epoch"] is not None
                    or current["train_c_values"]
                    or current["test_c_values"]
                    or current["train_miss_pairs"]
                    or current["test_miss_pairs"]
                ):
                    segments.append(finalize_segment(current))
                    current = new_segment()
                continue

            train_match = TRAIN_RE.match(line)
            if train_match:
                epoch_str, _dataset_idx, c_acc_str, r_acc_str = train_match.groups()
                current["latest_epoch"] = int(epoch_str)
                current["train_c_values"].append(to_float(c_acc_str))
                current["train_r_values"].append(to_float(r_acc_str))
                continue

            test_match = TEST_RE.match(line)
            if test_match:
                _dataset_idx, c_acc_str, r_acc_str = test_match.groups()
                current["test_c_values"].append(to_float(c_acc_str))
                current["test_r_values"].append(to_float(r_acc_str))
                continue

            train_miss_match = TRAIN_MISS_RE.match(line)
            if train_miss_match:
                miss_c_str, miss_r_str = train_miss_match.groups()
                current["train_miss_pairs"].append((int(miss_c_str), int(miss_r_str)))
                continue

            test_miss_match = TEST_MISS_RE.match(line)
            if test_miss_match:
                _dataset_idx, miss_c_str, miss_r_str = test_miss_match.groups()
                current["test_miss_pairs"].append((int(miss_c_str), int(miss_r_str)))
                continue

            loss_match = LOSS_RE.match(line)
            if loss_match:
                loss_value = to_float(loss_match.group(1))
                if isinstance(loss_value, float) and math.isnan(loss_value):
                    current["saw_nan_loss"] = True

    if (
        current["latest_epoch"] is not None
        or current["train_c_values"]
        or current["test_c_values"]
        or current["train_miss_pairs"]
        or current["test_miss_pairs"]
    ):
        segments.append(finalize_segment(current))

    if not segments:
        segments = [finalize_segment(new_segment())]

    for idx, seg in enumerate(segments):
        seg["subproblem_name"] = action_names[idx] if idx < len(action_names) else f"subproblem_{idx}"

    train_last_miss_c_total = sum(seg["train_last_miss_c"] or 0 for seg in segments)
    train_last_miss_r_total = sum(seg["train_last_miss_r"] or 0 for seg in segments)
    test_last_miss_c_total = sum(seg["test_last_miss_c"] or 0 for seg in segments)
    test_last_miss_r_total = sum(seg["test_last_miss_r"] or 0 for seg in segments)

    train_zero_subproblems = sum(1 for seg in segments if seg["last_train_success_zero_miss"])
    test_zero_subproblems = sum(1 for seg in segments if seg["last_test_success_zero_miss"])

    all_train_segments_ok = len(segments) > 0 and train_zero_subproblems == len(segments)
    all_test_segments_ok = len(segments) > 0 and test_zero_subproblems == len(segments)

    return {
        "model_bucket": model_bucket,
        "c_setting": c_setting,
        "task": task_name,
        "log_path": str(log_path),
        "subproblems": len(segments),
        "subproblem_names": [seg["subproblem_name"] for seg in segments],
        "train_zero_miss_subproblems": train_zero_subproblems,
        "test_zero_miss_subproblems": test_zero_subproblems,
        "epochs_seen": max((seg["epochs_seen"] or 0) for seg in segments) if segments else None,
        "train_last_c": mean_or_none([seg["train_last_c"] for seg in segments]),
        "train_best_c": mean_or_none([seg["train_best_c"] for seg in segments]),
        "train_last_r": mean_or_none([seg["train_last_r"] for seg in segments]),
        "train_best_r": mean_or_none([seg["train_best_r"] for seg in segments]),
        "test_last_c": mean_or_none([seg["test_last_c"] for seg in segments]),
        "test_best_c": mean_or_none([seg["test_best_c"] for seg in segments]),
        "test_last_r": mean_or_none([seg["test_last_r"] for seg in segments]),
        "test_best_r": mean_or_none([seg["test_best_r"] for seg in segments]),
        "test_evals": sum(seg["test_evals"] for seg in segments),
        "train_last_miss_c": train_last_miss_c_total,
        "train_last_miss_r": train_last_miss_r_total,
        "test_last_miss_c": test_last_miss_c_total,
        "test_last_miss_r": test_last_miss_r_total,
        "last_train_success_zero_miss": all_train_segments_ok,
        "last_test_success_zero_miss": all_test_segments_ok,
        "nan_loss": any(seg["nan_loss"] for seg in segments),
        "segments": segments,
    }


def summarize_group(rows: list[dict], include_failed: bool) -> dict:
    if include_failed:
        selected = rows
    else:
        selected = [row for row in rows if row["test_last_c"] is not None]

    test_last_c = [row["test_last_c"] for row in selected if row["test_last_c"] is not None]
    test_best_c = [row["test_best_c"] for row in selected if row["test_best_c"] is not None]
    train_last_c = [row["train_last_c"] for row in selected if row["train_last_c"] is not None]

    def mean_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "runs": len(rows),
        "runs_used": len(selected),
        "failed_or_incomplete": len(rows) - len(selected),
        "mean_test_last_c": mean_or_none(test_last_c),
        "mean_test_best_c": mean_or_none(test_best_c),
        "mean_train_last_c": mean_or_none(train_last_c),
        "nan_loss_runs": sum(1 for row in rows if row["nan_loss"]),
    }


def latest_rows_by_task(rows: list[dict]) -> list[dict]:
    latest_by_task = {}
    for row in rows:
        key = (row["model_bucket"], row["c_setting"], row["task"])
        current = latest_by_task.get(key)
        if current is None or row["log_path"] > current["log_path"]:
            latest_by_task[key] = row
    return [latest_by_task[key] for key in sorted(latest_by_task)]


def print_overview(parsed_rows: list[dict], include_failed: bool) -> None:
    print(f"Found {len(parsed_rows)} run log(s).")
    if not parsed_rows:
        return

    grouped = defaultdict(list)
    for row in parsed_rows:
        grouped[(row["model_bucket"], row["c_setting"])].append(row)

    print("\n=== Aggregate by model/c-setting ===")
    for (model_bucket, c_setting) in sorted(grouped):
        group_rows = grouped[(model_bucket, c_setting)]
        summary = summarize_group(group_rows, include_failed)
        latest_group_rows = latest_rows_by_task(group_rows)
        train_zero_tasks = sum(1 for row in latest_group_rows if row["last_train_success_zero_miss"])
        test_zero_tasks = sum(1 for row in latest_group_rows if row["last_test_success_zero_miss"])

        train_success_subproblems = []
        test_success_subproblems = []
        for row in latest_group_rows:
            task = row["task"]
            for seg in row.get("segments", []):
                subproblem_name = seg.get("subproblem_name", "unknown")
                label = f"{task}:{subproblem_name}"
                if seg.get("last_train_success_zero_miss", False):
                    train_success_subproblems.append(label)
                if seg.get("last_test_success_zero_miss", False):
                    test_success_subproblems.append(label)

        train_success_list = ", ".join(train_success_subproblems) if train_success_subproblems else "-"
        test_success_list = ", ".join(test_success_subproblems) if test_success_subproblems else "-"
        print(
            f"{model_bucket} / {c_setting}: "
            f"runs={summary['runs']} "
            f"used={summary['runs_used']} "
            f"failed={summary['failed_or_incomplete']} "
            f"tasks={len(latest_group_rows)} "
            f"train_zero_miss_tasks={train_zero_tasks} "
            f"test_zero_miss_tasks={test_zero_tasks} "
            f"nan_loss={summary['nan_loss_runs']} "
            f"mean_test_last_c={float_to_str(summary['mean_test_last_c'])} "
            f"mean_test_best_c={float_to_str(summary['mean_test_best_c'])}"
        )
        print(f"  train_success_subproblems=[{train_success_list}]")
        print(f"  test_success_subproblems=[{test_success_list}]")

    print("\n=== Per task (latest run per model/c/task) ===")
    for row in latest_rows_by_task(parsed_rows):
        model_bucket = row["model_bucket"]
        c_setting = row["c_setting"]
        task = row["task"]
        print(
            f"{model_bucket} / {c_setting} / {task}: "
            f"subproblems={row['subproblems']} "
            f"epochs={row['epochs_seen']} "
            f"train_last_c={float_to_str(row['train_last_c'])} "
            f"test_last_c={float_to_str(row['test_last_c'])} "
            f"train_last_miss=({row['train_last_miss_c']},{row['train_last_miss_r']}) "
            f"test_last_miss=({row['test_last_miss_c']},{row['test_last_miss_r']}) "
            f"train_ok={'yes' if row['last_train_success_zero_miss'] else 'no'} "
            f"test_ok={'yes' if row['last_test_success_zero_miss'] else 'no'} "
            f"test_best_c={float_to_str(row['test_best_c'])} "
            f"nan_loss={'yes' if row['nan_loss'] else 'no'}"
        )
        print(f"    names={', '.join(row.get('subproblem_names', []))}")


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        raise SystemExit(f"Outputs directory does not exist: {outputs_dir}")

    log_paths = sorted(outputs_dir.glob("**/output_*.log"))
    parsed_rows = [parse_log_file(path, outputs_dir) for path in log_paths]

    print_overview(parsed_rows, args.include_failed)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "outputs_dir": str(outputs_dir),
            "runs": parsed_rows,
        }
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote JSON summary to: {args.json_output}")


if __name__ == "__main__":
    main()
