import argparse
import csv
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CSV_PATHS = [
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786878932/benchmark.csv", # NDLM 6L
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786878936/benchmark.csv", # NLM 6L
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786879480/benchmark.csv", # NLM 4L
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402763/benchmark.csv", # NDLM 4l H10
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786880040/benchmark.csv", # NDLM 4L
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881318/benchmark.csv", # NDLM 4L H5
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881321/benchmark.csv", # NLM 4L H5
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881718/benchmark.csv", # NLM L6 H5
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1786881723/benchmark.csv", # NDLM 6L H5
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401509/benchmark.csv", # NDLM 4L H1
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401510/benchmark.csv", # NDLM 4L H2
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401581/benchmark.csv", # NDLM 4L H3
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401617/benchmark.csv", # NLM 4L H4
#     # "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787401927/benchmark.csv", # NDLM 4L H4
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402164/benchmark.csv", # NLM 4L H4
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402197/benchmark.csv", # NLM 4L H3
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402243/benchmark.csv", # NLM 4L H1
#     "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark/benchmark_1787402239/benchmark.csv", # NLM 4L H2
# ]

CSV_PATHS = [
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark_logistic_dimensions/benchmark_1787404937/benchmark.csv",
    "/work/rleap1/jan.dornhege/NDLM/outputs/benchmark_logistic_dimensions/benchmark_1787404928/benchmark.csv",
]

OUTPUT_DIR = "/work/rleap1/jan.dornhege/NDLM/plots"
OUTPUT_PATH = str(Path(OUTPUT_DIR) / "benchmark_time_memory_vs_objects.png")


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

    line_styles = cycle(["-", "--", "-.", ":"])
    markers = cycle(["o", "s", "^", "D", "v", "P", "*", "X"])

    if args.include_time:
        fig, (ax_time, ax_mem) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("Benchmark comparison: time and CUDA peak memory vs. objects")

        for (model_name, num_layers, hidden_size), rows in by_config.items():
            x = [row["num_objects"] for row in rows]
            y_time = [row["time_s"] for row in rows]
            y_mem = [row["cuda_peak_mb"] for row in rows]
            label = f"{model_name} (layers={num_layers}, hidden={hidden_size})"

            ax_time.plot(
                x,
                y_time,
                marker=next(markers),
                linestyle=next(line_styles),
                linewidth=2,
                label=label,
            )
            ax_mem.plot(
                x,
                y_mem,
                marker=next(markers),
                linestyle=next(line_styles),
                linewidth=2,
                label=label,
            )

        ax_time.set_ylabel("Avg inference time (s)")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")

        ax_mem.set_xlabel("Num objects")
        ax_mem.set_ylabel("Peak CUDA memory (MB)")
        # ax_mem.set_yscale("log")
        ax_mem.grid(True, alpha=0.3)
        ax_mem.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")

    else:
        fig, ax_mem = plt.subplots(1, 1, figsize=(14, 7))

        for (model_name, num_layers, hidden_size), rows in by_config.items():
            x = [row["num_objects"] for row in rows]
            y_mem = [row["cuda_peak_mb"] for row in rows]
            label = f"{model_name} (layers={num_layers}, hidden={hidden_size})"

            ax_mem.plot(
                x,
                y_mem,
                marker=next(markers),
                linestyle=next(line_styles),
                linewidth=3,
                label=label,
            )

        ax_mem.set_xlabel("Num objects")
        ax_mem.set_ylabel("Peak CUDA memory (MB)")
        # ax_mem.set_yscale("log")
        ax_mem.grid(True, alpha=0.3)
        ax_mem.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
