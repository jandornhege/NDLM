import os
import re
from collections import defaultdict

ROOT = "outputs/1D_ARC/NLM_B2_all"  # change this

results = defaultdict(lambda: {
    "total": 0,
    "train_zero": 0,
    "test_zero": 0,
    "both_zero": 0
})

# regex for the final result lines
train_pattern = re.compile(
    r"Final Train Results.*?c_miss': (\d+)"
)
test_pattern = re.compile(
    r"Final Test Results.*?c_miss': (\d+)"
)

for cx in os.listdir(ROOT):
    cx_path = os.path.join(ROOT, cx)

    if not os.path.isdir(cx_path) or not cx.startswith("c"):
        continue

    for task in os.listdir(cx_path):
        task_path = os.path.join(cx_path, task)

        if not os.path.isdir(task_path):
            continue

        # find timestamp experiment directory
        exp_dirs = [
            d for d in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, d))
        ]

        if len(exp_dirs) == 0:
            continue

        exp_path = os.path.join(task_path, exp_dirs[0])
        log_file = os.path.join(exp_path, "output.log")

        if not os.path.isfile(log_file):
            continue

        with open(log_file, "r") as f:
            log = f.read()

        train_match = train_pattern.search(log)
        test_match = test_pattern.search(log)

        # Prefer the final results from output.log
        if train_match is not None:
            train_miss = int(train_match.group(1))
        else:
            train_miss = None

        if test_match is not None:
            test_miss = int(test_match.group(1))
        else:
            test_miss = None

        # If one of them is missing, fall back to output_act.log
        if train_miss is None or test_miss is None:
            act_log = os.path.join(exp_path, "output_act.log")

            if os.path.isfile(act_log):
                with open(act_log, "r") as f:
                    act_log_text = f.read()

                # Last train misclassification line
                if train_miss is None:
                    train_matches = re.findall(
                        r"Total Misclassifications:\s*\((\d+),",
                        act_log_text,
                    )
                    if train_matches:
                        train_miss = min(map(int, train_matches))

                # Last test evaluation
                if test_miss is None:
                    test_matches = re.findall(
                        r"Test Misclassifications.*?c:(\d+)",
                        act_log_text,
                    )
                    if test_matches:
                        test_miss = int(test_matches[-1])

        # Skip if we still couldn't recover them
        if train_miss is None or test_miss is None:
            print(f"Could not recover results for {task_path}")
            continue
        results[cx]["total"] += 1

        if train_miss == 0:
            results[cx]["train_zero"] += 1

        if test_miss == 0:
            results[cx]["test_zero"] += 1

        if train_miss == 0 and test_miss == 0:
            results[cx]["both_zero"] += 1


# print results
for cx, r in sorted(results.items()):
    print(
        f"{cx}: "
        f"total={r['total']}, "
        f"train=0: {r['train_zero']}, "
        f"test=0: {r['test_zero']}, "
        f"both=0: {r['both_zero']}"
    )