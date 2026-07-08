# NDLM

Neural Description Logic Machines.

## Quick Start

### 1. Install NDLM in editable mode

From the repository root:

```bash
pip install -e .
```

### 2. Add Neural Logic Machines to PYTHONPATH

NDLM depends on the Google Neural Logic Machines repository:
https://github.com/google/neural-logic-machines/tree/master

Set the path to your local clone:

```bash
export PYTHONPATH=/path/to/neural-logic-machines:/path/to/neural-logic-machines/third_party/Jacinle:$PYTHONPATH
```

### 3. Run experiments

```bash
cd src/tasks
```

Action Model Learning:

```bash
python learn_task.py --task ActionModel --name DOMAIN_NAME --model NDLM
python learn_task.py --task ActionModel --name DOMAIN_NAME --model NLM
```

1D-ARC Learning:

```bash
python learn_task.py --task 1DARC --name TASK_NAME --model NDLM
python learn_task.py --task 1DARC --name TASK_NAME --model NLM
```

Change model parameters in configs.py

