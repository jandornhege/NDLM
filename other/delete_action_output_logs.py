#!/usr/bin/env python3
"""Delete per-action output_*.log files while preserving output.log.

Rules per directory:
- Keep output.log (if present).
- Delete files matching output_*.log (e.g. output_move.log, output_123.log).

Usage:
  python delete_action_output_logs.py /path/to/NDLM/outputs
  python delete_action_output_logs.py /path/to/NDLM/outputs --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def collect_deletions(root: Path) -> list[Path]:
    to_delete: list[Path] = []

    for directory in [p for p in root.rglob("*") if p.is_dir()]:
        action_logs = [
            f
            for f in directory.iterdir()
            if f.is_file()
            and f.suffix == ".log"
            and f.name.startswith("output_")
            and f.name != "output.log"
        ]
        if action_logs:
            to_delete.extend(action_logs)

    return sorted(to_delete)


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

    deletions = collect_deletions(root)
    bytes_to_free = sum(p.stat().st_size for p in deletions if p.exists())

    if not deletions:
        print("No files matched the deletion rule.")
        return 0

    print(
        f"Matched {len(deletions)} file(s), reclaimable space: "
        f"{format_bytes(bytes_to_free)} ({bytes_to_free} bytes)"
    )
    for file_path in deletions:
        print(file_path)

    if not args.execute:
        print("\nDry run only. Re-run with --execute to delete these files.")
        return 0

    deleted = 0
    freed_bytes = 0
    for file_path in deletions:
        if file_path.exists():
            freed_bytes += file_path.stat().st_size
        file_path.unlink(missing_ok=True)
        deleted += 1

    print(
        f"\nDeleted {deleted} file(s), freed {format_bytes(freed_bytes)} "
        f"({freed_bytes} bytes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
