#!/usr/bin/env python3
"""Evaluate the saved best checkpoint for each CV fold and aggregate accuracy across folds.

This script does not retrain models. It scans the saved NDLM CV runs under
NDLM/outputs/GRAPH_EXPERIMENTS, loads the best checkpoint from each fold, runs the
trainer.test(load_best_model=True) path, and reports the fold-wise accuracies plus a
per-domain mean ± variance summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NDLM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = NDLM_ROOT.parent
GRAPH_ROOT = WORKSPACE_ROOT / "ProvablyPowerfulGraphNetworks_torch"
DEFAULT_ROOT = WORKSPACE_ROOT / "NDLM" / "outputs" / "GRAPH_EXPERIMENTS_new_configs_cv"

sys.path.insert(0, str(NDLM_ROOT / "src"))
sys.path.insert(0, str(GRAPH_ROOT))

from data_loader.data_generator import DataGenerator
from trainers.trainer import Trainer


def _load_run_config(run_dir: Path):
    config_path = run_dir / "summary" / "used_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing used_config.json in {run_dir}")
    with config_path.open() as f:
        return json.load(f)


def _resolve_device(device: str | None):
    requested = (device or "auto").lower()
    if requested in ("", "auto", "none"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cpu"):
        return "cpu"
    try:
        torch.device(requested)
        return requested
    except (TypeError, ValueError):
        return "cuda" if torch.cuda.is_available() else "cpu"


def _best_epoch_from_summary(summary_dir: Path):
    csv_path = summary_dir / "per_epoch_stats.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "val_accuracy" not in df.columns:
        return None
    return int(df["val_accuracy"].idxmax())


def collect_cv_epoch_rows(root: Path, dataset: str | None = None):
    """Collect epoch-wise validation accuracies from each saved CV fold."""
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "summary" / "per_epoch_stats.csv"
        if not csv_path.exists():
            continue
        try:
            config = _load_run_config(run_dir)
        except Exception:
            continue
        d = str(config.get("dataset_name", "")).upper()
        if dataset is not None and d != dataset.upper():
            continue
        if config.get("protocol") != "cv" and "cv" not in run_dir.name.lower():
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "epoch" not in df.columns or "val_accuracy" not in df.columns:
            continue
        fold_num = int(config.get("num_fold", 0) or 0)
        if fold_num == 0:
            continue
        subset = df[["epoch", "val_accuracy"]].copy()
        subset["dataset"] = d
        subset["fold"] = fold_num
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["epoch", "val_accuracy", "dataset", "fold"])


def summarize_cv_epoch_accuracy(rows: pd.DataFrame):
    """Pick the epoch with the highest mean validation accuracy across folds, matching the paper's CV rule."""
    if rows.empty:
        return None
    agg = rows.groupby("epoch", as_index=False)["val_accuracy"].agg(["mean", "std"])
    if agg.empty:
        return None
    agg = agg.rename(columns={"mean": "mean_accuracy", "std": "std_accuracy"})
    best_epoch = int(agg["mean_accuracy"].idxmax())
    best_row = agg.loc[agg["mean_accuracy"].idxmax()]
    return {
        "best_epoch": best_epoch,
        "mean_accuracy": float(best_row["mean_accuracy"]),
        "std_accuracy": float(best_row["std_accuracy"]),
        "variance_accuracy": float(best_row["std_accuracy"] ** 2),
        "mean_plus_minus_std": f"{best_row['mean_accuracy']:.4f} ± {best_row['std_accuracy']:.4f}",
    }


def _build_ndlm_model_wrapper():
    import importlib.util

    module_path = NDLM_ROOT / "src" / "tasks" / "graph_10fold_experiment.py"
    spec = importlib.util.spec_from_file_location("ndlm_graph_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NDLMModelWrapper


def run_fold_test(run_dir: Path, device: str | None = None):
    config_dict = _load_run_config(run_dir)
    config = config_dict.copy()
    config["device"] = _resolve_device(device or config.get("device"))
    config["parent_dir"] = str(run_dir)
    config["summary_dir"] = str(run_dir / "summary")
    config["checkpoint_dir"] = str(run_dir / "checkpoint")

    checkpoint_path = Path(config["checkpoint_dir"]) / "best.tar"
    print(f"Loading best checkpoint from: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best checkpoint found in {config['checkpoint_dir']}")

    print(f"Loading best checkpoint from: {checkpoint_path}")

    sys.path.insert(0, str(NDLM_ROOT / "src"))
    sys.path.insert(0, str(GRAPH_ROOT))
    from ndlm.configs import config_object
    from torch import nn

    ndlm_cfg = config_object()
    ndlm_cfg.NUM_LAYERS = int(config.get("ndlm", {}).get("num_layers", config.get("num_layers", 4)))
    ndlm_cfg.NUM_HIDDEN_CONCEPTS = int(config.get("ndlm", {}).get("hidden_concepts", config.get("hidden_concepts", 10)))
    ndlm_cfg.NUM_HIDDEN_ROLES = int(config.get("ndlm", {}).get("hidden_roles", config.get("hidden_roles", 10)))
    ndlm_cfg.MODE = str(config.get("ndlm", {}).get("mode", config.get("mode", "strict")))
    act = str(config.get("ndlm", {}).get("activation", config.get("activation", "identity")))
    ndlm_cfg.ACTIVATION_FUNCTION = nn.Identity() if act == "identity" else nn.Sigmoid()
    ndlm_cfg.TRANSITIVE_CLOSURE = bool(config.get("ndlm", {}).get("transitive_closure", config.get("transitive_closure", False)))
    ndlm_cfg.RESIDUAL = bool(config.get("ndlm", {}).get("residual", config.get("residual", False)))
    ndlm_cfg.INPUT_RESIDUAL = bool(config.get("ndlm", {}).get("input_residual", config.get("input_residual", False)))
    ndlm_cfg.INITIAL_FFN = bool(config.get("ndlm", {}).get("initial_ffn", config.get("initial_ffn", False)))

    device = torch.device(config["device"])

    # The dataloader expects a config-like object with attribute access.
    class EasyConfig(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    cfg = EasyConfig(config)
    cfg.hyperparams = EasyConfig(cfg.get("hyperparams", {}))
    cfg.num_fold = cfg.get("num_fold", 0)
    cfg.ndlm = EasyConfig(cfg.get("ndlm", {}))
    data = DataGenerator(cfg)
    model_wrapper_cls = _build_ndlm_model_wrapper()
    wrapper = model_wrapper_cls(cfg, ndlm_cfg, device)
    trainer = Trainer(wrapper, data, cfg)
    trainer.best_epoch = _best_epoch_from_summary(run_dir / "summary") or 0
    metric = trainer.test(load_best_model=True)
    acc = float(metric[0])
    loss = float(metric[1])
    return {
        "dataset": str(cfg.get("dataset_name", run_dir.name)).upper(),
        "run_dir": str(run_dir),
        "fold": int(cfg.get("num_fold", 0) or 0),
        "best_epoch": trainer.best_epoch,
        "test_accuracy": acc,
        "test_loss": loss,
    }


def discover_cv_runs(root: Path, dataset: str | None = None):
    candidates = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        used_config_path = run_dir / "summary" / "used_config.json"
        checkpoint_path = run_dir / "checkpoint" / "best.tar"
        if not used_config_path.exists() or not checkpoint_path.exists():
            continue
        try:
            config = _load_run_config(run_dir)
        except Exception:
            continue
        d = str(config.get("dataset_name", "")).upper()
        if dataset is not None and d != dataset.upper():
            continue
        if config.get("protocol") != "cv" and "cv" not in run_dir.name.lower():
            continue
        candidates.append((run_dir, config))
    return candidates


def summarize_by_domain(results):
    summary = []
    by_domain = {}
    for item in results:
        by_domain.setdefault(item["dataset"], []).append(item["test_accuracy"])

    for domain in sorted(by_domain):
        values = np.asarray(by_domain[domain], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        var = float(values.var(ddof=0))
        summary.append({
            "dataset": domain,
            "fold_count": int(len(values)),
            "mean_accuracy": mean,
            "std_accuracy": std,
            "variance_accuracy": var,
            "mean_plus_minus_std": f"{mean:.4f} ± {std:.4f}",
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate CV validation curves across folds using the paper's protocol: pick the epoch with the highest mean validation accuracy across folds.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root directory containing the saved graph CV runs.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional single dataset (e.g. COLLAB).")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for the summary table.")
    args = parser.parse_args()

    rows = collect_cv_epoch_rows(args.root, dataset=args.dataset)
    if rows.empty:
        print(f"No CV per_epoch stats found under {args.root}; falling back to per-fold checkpoint evaluation.")
        runs = discover_cv_runs(args.root, dataset=args.dataset)
        if not runs:
            print(f"No CV run directories with best.tar checkpoints found under {args.root}")
            return 1

        results = []
        for run_dir, config in runs:
            fold_num = int(config.get("num_fold", 0) or 0)
            if fold_num == 0:
                print(f"Skipping fold 0 run: {run_dir}")
                continue
            try:
                result = run_fold_test(run_dir, device=args.device)
                results.append(result)
                print(f"{result['dataset']} fold={result['fold']} best_epoch={result['best_epoch']} test_acc={result['test_accuracy']:.4f} test_loss={result['test_loss']:.4f}")
            except Exception as exc:
                print(f"FAILED {run_dir}: {exc}", file=sys.stderr)

        if not results:
            print("No fold evaluations succeeded.")
            return 1

        summary = summarize_by_domain(results)
        print("\nDOMAIN SUMMARY")
        print("dataset | folds | mean_accuracy | std | variance | mean ± std")
        for row in summary:
            print(f"{row['dataset']:<12} | {row['fold_count']:>5} | {row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['variance_accuracy']:.6f} | {row['mean_plus_minus_std']}")

        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary).to_csv(args.csv, index=False)
            print(f"\nSaved summary CSV to {args.csv}")
        return 0

    summary = summarize_cv_epoch_accuracy(rows)
    if summary is None:
        print("No usable val_accuracy rows available for CV epoch aggregation.")
        return 1

    dataset_summary = []
    by_dataset = rows.groupby("dataset")
    for dataset_name in sorted(by_dataset.groups):
        ds_rows = by_dataset.get_group(dataset_name)
        agg = ds_rows.groupby("epoch", as_index=False)["val_accuracy"].agg(["mean", "std"])
        agg = agg.rename(columns={"mean": "mean_accuracy", "std": "std_accuracy"})
        best_epoch = int(agg["mean_accuracy"].idxmax())
        best_row = agg.loc[best_epoch]
        dataset_summary.append({
            "dataset": dataset_name,
            "fold_count": int(ds_rows["fold"].nunique()),
            "best_epoch": best_epoch,
            "mean_accuracy": float(best_row["mean_accuracy"]),
            "std_accuracy": float(best_row["std_accuracy"]),
            "variance_accuracy": float(best_row["std_accuracy"] ** 2),
            "mean_plus_minus_std": f"{best_row['mean_accuracy']:.4f} ± {best_row['std_accuracy']:.4f}",
        })

    print("\nCV EPOCH SUMMARY")
    print("dataset | folds | best_epoch | mean_accuracy | std | variance | mean ± std")
    for row in dataset_summary:
        print(f"{row['dataset']:<12} | {row['fold_count']:>5} | {row['best_epoch']:>10} | {row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['variance_accuracy']:.6f} | {row['mean_plus_minus_std']}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dataset_summary).to_csv(args.csv, index=False)
        print(f"\nSaved summary CSV to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
