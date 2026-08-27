import os
import re
from pprint import pprint

def plot_results(results):
    import matplotlib.pyplot as plt

    for num_layers, values in results.items():
        line_nums = sorted(values.keys())
        c_miss_values = [values[line_num] for line_num in line_nums]

        plt.plot(line_nums, c_miss_values, label=f"L{num_layers}")

    plt.xlabel("Line Number")
    plt.ylabel("c_miss")
    plt.title("c_miss vs Line Number for Different Layer Configurations")
    plt.legend()
    plt.grid()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}.png")
name = "NLM_correct_B3_sig"
NAME = "NLM_correct/B3_sig"
ROOT = "outputs/hard_graphs/LINE/"     
ROOT = os.path.join(ROOT, NAME)  
# Matches:
# Final Train Results for action line_12: ... 'c_miss': 16
pattern = re.compile(
    r"Final Train Results for action line_(\d+):.*?'c_miss':\s*(\d+)"
)

results = {}

for dirname in sorted(os.listdir(ROOT)):
    exp_dirs = os.listdir(os.path.join(ROOT, dirname))
    values = {}
    for exp_dir in exp_dirs:
        print(exp_dir)
        m = re.fullmatch(r"L(\d+)", dirname)
        if not m:
            continue

        num_layers = int(m.group(1))
        logfile = os.path.join(ROOT, dirname, exp_dir, "output.log")

        if not os.path.exists(logfile):
            print(f"Log file does not exist: {logfile}")
            continue


        with open(logfile) as f:
            for line in f:
                
                m = pattern.search(line)
                if m:
                    line_num = int(m.group(1))
                    c_miss = int(m.group(2))
                    values[line_num] = c_miss

    results[num_layers] = values

pprint(results)

plot_results(results)
