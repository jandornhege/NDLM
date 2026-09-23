import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch

DEFAULT_ROOT = Path(
    os.environ.get(
        "GRAPH_EXPERIMENTS_ROOT",
        "/work/rleap1/jan.dornhege/NDLM/outputs/GRAPH_EXPERIMENTS_newer_configs",
    )
)
root = DEFAULT_ROOT

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
        "run_dir": csv_path.parent.parent,
    }


def load_used_config(run_dir: Path):
    config_path = run_dir / "summary" / "used_config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        return json.load(f)


def get_best_by_domain(domain_filter=None):
    by_domain = {}
    for csv_path in sorted(root.glob("*/summary/per_epoch_stats.csv")):
        run = parse_best_run(csv_path)
        if run is None:
            continue

        domain = infer_domain(csv_path.parent.parent)
        if domain_filter is not None and domain.upper() != domain_filter.upper():
            continue
        by_domain.setdefault(domain, []).append(run)

    return {
        domain: sorted(runs, key=lambda x: x["val_accuracy"], reverse=True)
        for domain, runs in by_domain.items()
    }


def resolve_device_name(device_value=None):
    requested = str(device_value).lower() if device_value is not None else "auto"
    if requested in ("", "auto", "None", "none"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print(f"Warning: requested device '{requested}' but CUDA is unavailable; falling back to CPU.")
        return "cpu"

    if requested.startswith("cpu"):
        return "cpu"

    try:
        torch.device(requested)
        return requested
    except (TypeError, ValueError):
        print(f"Warning: unsupported device '{requested}'; falling back to CPU.")
        return "cpu"


def build_ndlm_config(config_dict):
    sys.path.insert(0, str(Path("/work/rleap1/jan.dornhege/NDLM/src")))
    sys.path.insert(0, str(Path("/work/rleap1/jan.dornhege/ProvablyPowerfulGraphNetworks_torch")))

    from ndlm.configs import config_object
    from torch import nn

    ndlm_cfg = config_object()
    data = config_dict.get("ndlm", {})
    ndlm_cfg.NUM_LAYERS = int(data.get("num_layers", 4))
    ndlm_cfg.NUM_HIDDEN_CONCEPTS = int(data.get("hidden_concepts", 10))
    ndlm_cfg.NUM_HIDDEN_ROLES = int(data.get("hidden_roles", 10))
    ndlm_cfg.MODE = data.get("mode", "strict")
    act = data.get("activation", "identity")
    ndlm_cfg.ACTIVATION_FUNCTION = nn.Identity() if act == "identity" else nn.Sigmoid()
    ndlm_cfg.TRANSITIVE_CLOSURE = bool(data.get("transitive_closure", False))
    ndlm_cfg.RESIDUAL = bool(data.get("residual", False))
    ndlm_cfg.INPUT_RESIDUAL = bool(data.get("input_residual", False))
    ndlm_cfg.INITIAL_FFN = bool(data.get("initial_ffn", False))
    return ndlm_cfg


def run_best_test(best_run):
    config_dict = load_used_config(best_run["run_dir"])
    if config_dict is None:
        raise FileNotFoundError(f"No used_config.json found for {best_run['config']}")

    sys.path.insert(0, str(Path("/work/rleap1/jan.dornhege/NDLM/src")))
    sys.path.insert(0, str(Path("/work/rleap1/jan.dornhege/ProvablyPowerfulGraphNetworks_torch")))

    import importlib.util
    try:
        from easydict import EasyDict
    except ModuleNotFoundError:
        class EasyDict(dict):
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

            def __setattr__(self, key, value):
                self[key] = value

    from data_loader.data_generator import DataGenerator
    from trainers.trainer import Trainer

    module_path = Path('/work/rleap1/jan.dornhege/NDLM/src/tasks/graph_10fold_experiment.py')
    spec = importlib.util.spec_from_file_location('ndlm_graph_runner', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    NDLMModelWrapper = module.NDLMModelWrapper

    config = EasyDict(config_dict)
    config.num_fold = config.get("num_fold", 0)
    config.device = resolve_device_name(config.get("device"))
    config.parent_dir = str(best_run["run_dir"])
    config.summary_dir = str(best_run["run_dir"] / "summary")
    config.checkpoint_dir = str(best_run["run_dir"] / "checkpoint")

    checkpoint_path = Path(config.checkpoint_dir) / 'best.tar'
    if not checkpoint_path.exists():
        print(f"\nSkipping final test for {best_run['config']}: no best.tar checkpoint exists under {config.checkpoint_dir}.")
        return None

    ndlm_cfg = build_ndlm_config(config)
    device = torch.device(config.device)
    data = DataGenerator(config)
    wrapper = NDLMModelWrapper(config, ndlm_cfg, device)
    trainer = Trainer(wrapper, data, config)
    trainer.best_epoch = int(best_run["epoch"])

    print(f"\nFinal evaluation for {best_run['config']}")
    print(f"Selected epoch: {best_run['epoch']}, val_acc: {best_run['val_accuracy']:.4f}")
    print("Benchmark graph runs do not expose a separate public test split; the final test path reuses the selected validation split.")
    metric = trainer.test(load_best_model=True)
    print(f"Final metric: {metric}")
    return metric


def print_rankings(by_domain):
    print("BEST CONFIGS BY DOMAIN")
    print("=" * 140)

    for domain in sorted(by_domain):
        runs = by_domain[domain]
        print(f"\n[{domain}]")
        print(f"{'RANK':>4} | {'CONFIG':<90} | {'EPOCH':>5} | {'VAL_ACC':>8} | {'VAL_LOSS':>9} | {'TRAIN_ACC':>10}")
        print("-" * 140)

        for i, row in enumerate(runs[:min(args.top_n, len(runs))], start=1):
            print(
                f"{i:>4} | {row['config']:<90} | "
                f"{row['epoch']:>5} | {row['val_accuracy']:>8.4f} | "
                f"{row['val_loss']:>9.4f} | {row['train_accuracy']:>10.4f}"
            )


def main():
    global args, root
    parser = argparse.ArgumentParser(description="Rank graph results and optionally run the final eval on the best config.")
    parser.add_argument("root", nargs="?", default=None, help="Root directory containing experiment folders, e.g. outputs/GRAPH_EXPERIMENTS_NLM.")
    parser.add_argument("--root", dest="root_flag", default=None, help="Explicit root directory; equivalent to the positional argument.")
    parser.add_argument("--domain", default=None, help="Filter to one dataset/domain, e.g. COLLAB or PROTEINS.")
    parser.add_argument("--test-best", action="store_true", help="Run trainer.test() on the best config after ranking.")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top configs to print.")
    args = parser.parse_args()

    root = Path(args.root_flag or args.root or os.environ.get("GRAPH_EXPERIMENTS_ROOT", str(DEFAULT_ROOT))).expanduser()

    by_domain = get_best_by_domain(args.domain)
    if not by_domain:
        print(f"No valid per_epoch_stats.csv entries found under {root}")
        return

    print_rankings(by_domain)

    if args.test_best:
        for domain in sorted(by_domain):
            best = by_domain[domain][0]
            run_best_test(best)
            break


if __name__ == "__main__":
    main()