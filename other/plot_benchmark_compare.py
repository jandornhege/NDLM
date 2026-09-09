import argparse
import csv
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATHS = [
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786878932/benchmark.csv", # NDLM 6L
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786878936/benchmark.csv", # NLM 6L
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786879480/benchmark.csv", # NLM 4L
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402763/benchmark.csv", # NDLM 4l H10
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786880040/benchmark.csv", # NDLM 4L
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881318/benchmark.csv", # NDLM 4L H5
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881321/benchmark.csv", # NLM 4L H5
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881718/benchmark.csv", # NLM L6 H5
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881723/benchmark.csv", # NDLM 6L H5
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401509/benchmark.csv", # NDLM 4L H1
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401510/benchmark.csv", # NDLM 4L H2
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401581/benchmark.csv", # NDLM 4L H3
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401617/benchmark.csv", # NLM 4L H4
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401927/benchmark.csv", # NDLM 4L H4
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402164/benchmark.csv", # NLM 4L H4
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402197/benchmark.csv", # NLM 4L H3
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402243/benchmark.csv", # NLM 4L H1
    # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402239/benchmark.csv", # NLM 4L H2
]

# CSV_PATHS = [
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark_logistic_dimensions/benchmark_1787404937/benchmark.csv",
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark_logistic_dimensions/benchmark_1787404928/benchmark.csv",
# ]

OUTPUT_DIR = "/work/rleap1/jan.dornhege/NDLM/plots"
OUTPUT_PATH = str(Path(OUTPUT_DIR) / "benchmark_time_memory_vs_objects.png")

MODEL_COLORS = {
    "NDLM": "#0072B2",
    "NLM": "#D55E00",
}
CONFIG_STYLES = [
    ("-", "o"),
    ("--", "s"),
    ("-.", "^"),
    (":", "D"),
]


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "model_name": row["model_name"],
                    "num_layers": int(row.get("num_layers", 0)),
                    "hidden_size": int(row.get("hidden_size", 0)),
                    "num_objects": int(row["num_objects"]),
                    "time_s": float(row["avg_inference_time_s"]),
                    "cuda_peak_mb": float(row["peak_cuda_allocated_mb"]),
                }
            )
    rows.sort(key=lambda r: (r["num_objects"], r["model_name"], r["num_layers"], r["hidden_size"]))
    rows = [row for row in rows if row["num_objects"] <= 256]  # Filter out large object counts for better visualization
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark plot generator")
    parser.add_argument(
        "--include-time",
        action="store_true",
        help="Include the average inference time subplot in addition to memory.",
    )
    args = parser.parse_args()

    by_config = {}
    for csv_path in CSV_PATHS:
        for row in load_rows(csv_path):
            key = (row["model_name"], row["num_layers"], row["hidden_size"])
            by_config.setdefault(key, []).append(row)

    if not by_config:
        raise ValueError("No benchmark rows were found in the configured CSV files.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    configs_by_model = {}
    for model_name, num_layers, hidden_size in by_config:
        configs_by_model.setdefault(model_name, []).append((num_layers, hidden_size))
    style_by_config = {
        (model_name, num_layers, hidden_size): CONFIG_STYLES[index % len(CONFIG_STYLES)]
        for model_name, configs in configs_by_model.items()
        for index, (num_layers, hidden_size) in enumerate(sorted(configs))
    }

    def plot_metric(axis, metric, ylabel):
        for key in sorted(by_config):
            model_name, num_layers, hidden_size = key
            rows = sorted(by_config[key], key=lambda row: row["num_objects"])
            linestyle, marker = style_by_config[key]
            axis.plot(
                [row["num_objects"] for row in rows],
                [row[metric] for row in rows],
                color=MODEL_COLORS.get(model_name, "#4D4D4D"),
                marker=marker,
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                linestyle=linestyle,
                linewidth=1.8,
                label=f"{model_name}: depth={num_layers}, width={hidden_size}",
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#BDBDBD", linewidth=0.6, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3, width=0.8)

    if args.include_time:
        fig, (ax_time, ax_mem) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
        plot_metric(ax_time, "time_s", "Average inference time (s)")
        plot_metric(ax_mem, "cuda_peak_mb", "Peak CUDA memory (MB)")
        ax_mem.set_xlabel("Num objects")
        ax_time.legend(title="Configuration", frameon=False, ncol=2, loc="best")

    else:
        fig, ax_mem = plt.subplots(1, 1, figsize=(8.5, 4.8))
        plot_metric(ax_mem, "cuda_peak_mb", "Peak CUDA memory (MB)")
        ax_mem.set_xlabel("Num objects")
        ax_mem.legend(title="Configuration", frameon=False, ncol=2, loc="best")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
