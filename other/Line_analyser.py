import os
import re
from pprint import pprint
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


def mean_confidence_interval(data, confidence=0.95):
    """Returns (mean, lower, upper)."""
    data = np.asarray(data, dtype=float)
    # mean = np.mean(data)
    mean = np.min(data)

    if len(data) <= 1:
        return mean, mean, mean

    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    h = sem * t.ppf((1 + confidence) / 2.0, len(data) - 1)
    return mean, mean - h, mean + h


def plot_results(results):
    plt.figure(figsize=(8, 6))

    for num_layers, values in sorted(results.items()):

        line_nums = sorted(values.keys())

        means = []
        lowers = []
        uppers = []

        for line_num in line_nums:
            mean, low, high = mean_confidence_interval(values[line_num])
            means.append(mean)
            lowers.append(low)
            uppers.append(high)

        means = np.array(means)
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        plt.plot(line_nums, means, label=f"L{num_layers}")
        plt.ylim(-5, 60)
        # plt.fill_between(line_nums, lowers, uppers, alpha=0.2)

    plt.xlabel("Line Number")
    plt.ylabel("c_miss")
    plt.title("Average c_miss vs Line Number")
    plt.grid(True)
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    plt.show()
    plt.close()


# name = "NLM_residual_B2_sig"
# NAME = "NLM_residual/B2_sig"

for config in os.listdir("outputs/hard_graphs/LINE"):
# for config in ["NLM_high_lr_correct"]:
    for exp_name in os.listdir(os.path.join("outputs/hard_graphs/LINE", config)):
        name = f"{config}_{exp_name.replace('/', '_')}"
        NAME = os.path.join(config, exp_name)

        ROOT = os.path.join("outputs/hard_graphs/LINE", NAME)

        pattern = re.compile(
            r"Final Train Results for action line_(\d+):.*?'c_miss':\s*(\d+)"
        )

        results = {}

        for dirname in sorted(os.listdir(ROOT)):
            m = re.fullmatch(r"L(\d+)", dirname)
            if not m:
                continue

            num_layers = int(m.group(1))
            if num_layers > 8:
                continue

            # line_num -> list of c_miss values from different runs
            values = defaultdict(list)

            for exp_dir in os.listdir(os.path.join(ROOT, dirname)):
                logfile = os.path.join(ROOT, dirname, exp_dir, "output.log")

                if not os.path.exists(logfile):
                    continue

                with open(logfile) as f:
                    for line in f:
                        m2 = pattern.search(line)
                        if m2:  # Only consider line numbers less than 100
                            line_num = int(m2.group(1))
                            c_miss = int(m2.group(2))
                            values[line_num].append(c_miss)

            results[num_layers] = values

        pprint(results)

        plot_results(results)