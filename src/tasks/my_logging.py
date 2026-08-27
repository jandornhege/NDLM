
import sys
import time
import os

def init_logger(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = open(log_file_path, 'a')
    sys.stderr = open(log_file_path, 'a')
    # log(f"Logging initialized. Log file: {log_file_path}")

def log(string, end= "\n"):
    string = str(string)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] \t {string}", end=end, flush=True)
    sys.stdout.flush()