#!/usr/bin/env python3
"""Delete redundant non-final .pt files based on matching final/epoch variants.

A file is treated as "final" if its stem contains the keyword "final" as a token,
for example: model_final.pt, model-final.pt, model.final.pt.

A non-final file is deleted if there is a final file in the same directory whose
base key (filename stem with the final token removed) matches.

If there is no matching final file for a base key, only the highest-epoch non-final
file is kept and the other non-final files for that base key are deleted.

Examples:
    checkpoint_epoch_1990.pt + checkpoint_final.pt -> delete checkpoint_epoch_1990.pt
    checkpoint_epoch_1970.pt, checkpoint_epoch_1980.pt (no final) -> keep 1980 only

Usage:
  python delete_orphan_nonfinal_pt.py /path/to/NDLM/outputs
  python delete_orphan_nonfinal_pt.py /path/to/NDLM/outputs --execute
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

FINAL_TOKEN_RE = re.compile(r"(?i)(^|[^a-z0-9])final([^a-z0-9]|$)")
REMOVE_FINAL_RE = re.compile(r"(?i)(^|[^a-z0-9])final([^a-z0-9]|$)")
REMOVE_EPOCH_RE = re.compile(r"(?i)(?:[_\-.])epoch(?:[_\-.])\d+$")
EPOCH_EXTRACT_RE = re.compile(r"(?i)(?:^|[_\-.])epoch(?:[_\-.])(\d+)$")
SEPARATOR_CLEAN_RE = re.compile(r"[_\-.]{2,}")


def has_final_token(stem: str) -> bool:
    return FINAL_TOKEN_RE.search(stem) is not None


def base_key(stem: str) -> str:
    # Remove token-like appearances of "final" and normalize separators.
    cleaned = REMOVE_FINAL_RE.sub(r"\1\2", stem)
    cleaned = REMOVE_EPOCH_RE.sub("", cleaned)
    cleaned = SEPARATOR_CLEAN_RE.sub("_", cleaned)
    cleaned = cleaned.strip("_-.")
    return cleaned.lower()


def extract_epoch(stem: str) -> int | None:
    match = EPOCH_EXTRACT_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def process_deletions(root: Path, execute: bool) -> tuple[int, int]:
    matched = 0
    bytes_total = 0

    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        pt_files = [f for f in directory.iterdir() if f.is_file() and f.suffix == ".pt"]
        if not pt_files:
            continue

        grouped: dict[str, list[Path]] = defaultdict(list)
        for f in pt_files:
            grouped[base_key(f.stem)].append(f)

        for _, files in grouped.items():
            final_files = [f for f in files if has_final_token(f.stem)]
            non_final_files = [f for f in files if not has_final_token(f.stem)]

            if final_files:
                candidates = non_final_files
            else:
                if len(non_final_files) <= 1:
                    continue

                keeper = max(
                    non_final_files,
                    key=lambda p: (
                        extract_epoch(p.stem) is not None,
                        extract_epoch(p.stem) if extract_epoch(p.stem) is not None else -1,
                        p.stem,
                    ),
                )
                candidates = [f for f in non_final_files if f != keeper]
            for file_path in candidates:
                file_size = file_path.stat().st_size if file_path.exists() else 0
                if execute:
                    file_path.unlink(missing_ok=True)
                matched += 1
                bytes_total += file_size
            print(f"Deleted {len(candidates)} file(s) in {directory}.")

    return matched, bytes_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to scan (e.g. /work/.../NDLM/outputs)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, only print a dry-run list.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Error: {root} is not a directory")

    matched, bytes_total = process_deletions(root, execute=args.execute)

    if matched == 0:
        print("No files matched the deletion rule.")
        return 0

    if args.execute:
        print(
            f"Deleted {matched} file(s), freed {format_bytes(bytes_total)} "
            f"({bytes_total} bytes)."
        )
    else:
        print(
            f"\nMatched {matched} file(s), reclaimable space: "
            f"{format_bytes(bytes_total)} ({bytes_total} bytes)"
        )
        print("Dry run only. Re-run with --execute to delete these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
