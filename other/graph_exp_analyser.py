from pathlib import Path
import csv
import re

root = Path("/work/rleap1/jan.dornhege/experiments")

DATASET_HINTS = [
    "MUTAG", "PROTEINS", "PTC", "NCI109", "NCI1", "IMDBBINARY",
    "IMDBMULTI", "REDDITBINARY", "REDDITMULTI", "COLLAB",
    "ENZYMES", "DD", "AIDS", "BZR", "COX2"
]

def infer_domain(path: Path) -> str:
    name = path.name
    for hint in DATASET_HINTS:
        if hint.upper() in name.upper():
            return hint.upper()

    m = re.search(r"([A-Za-z0-9]+)(?=202[0-9]|_ndlm|_strict|_baseline)", name)
    if m:
        return m.group(1).upper()

    return "UNKNOWN"

def parse_best_run(csv_path: Path):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    best = None
    for r in rows:
        try:
            item = {
                "epoch": int(float(r["epoch"])),
                "train_accuracy": float(r["train_accuracy"]),
                "val_loss": float(r["val_loss"]),
                "val_accuracy": float(r["val_accuracy"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if best is None or item["val_accuracy"] > best["val_accuracy"]:
            best = item

    if best is None:
        return None

    return {
        "config": csv_path.parent.parent.name,
        "epoch": best["epoch"],
        "val_accuracy": best["val_accuracy"],
        "val_loss": best["val_loss"],
        "train_accuracy": best["train_accuracy"],
    }

by_domain = {}

for csv_path in sorted(root.glob("*/summary/per_epoch_stats.csv")):
    run = parse_best_run(csv_path)
    if run is None:
        continue

    domain = infer_domain(csv_path.parent.parent)
    by_domain.setdefault(domain, []).append(run)

print("BEST CONFIGS BY DOMAIN")
print("=" * 140)

for domain in sorted(by_domain):
    runs = sorted(by_domain[domain], key=lambda x: x["val_accuracy"], reverse=True)

    print(f"\n[{domain}]")
    print(f"{'RANK':>4} | {'CONFIG':<90} | {'EPOCH':>5} | {'VAL_ACC':>8} | {'VAL_LOSS':>9} | {'TRAIN_ACC':>10}")
    print("-" * 140)

    for i, row in enumerate(runs, start=1):
        print(
            f"{i:>4} | {row['config']:<90} | "
            f"{row['epoch']:>5} | {row['val_accuracy']:>8.4f} | "
            f"{row['val_loss']:>9.4f} | {row['train_accuracy']:>10.4f}"
        )