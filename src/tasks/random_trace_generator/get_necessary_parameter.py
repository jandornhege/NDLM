import subprocess
import os

domain = "data/logistics/logistics.pddl"
instance = "data/logistics/logistics-1.pddl"
domain_abs = os.path.abspath(domain)
instance_abs = os.path.abspath(instance)
size = 100
print(type(size))
result = subprocess.run(
    [
        "/work/rleap1/jan.dornhege/envs/graph_separator/bin/python3.12",
        "test.py",
        "--get_schema",
        "-d", domain_abs,
        "-i", instance_abs,
        "-s", str(size)
    
    ],
    cwd="../graph_separator",   # working directory for the script
    capture_output=True,
    text=True
)

print(result.stdout)