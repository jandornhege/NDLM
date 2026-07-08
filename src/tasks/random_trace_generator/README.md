# State Transition Generator

Generates all reachable state transitions for a specific PDDL action schema using Mimir.

## Requirements

- Python 3.12+
- pymimir (install via `pip install pymimir`)

## Usage

Run with your conda environment activated:

```bash
# Activate your environment (e.g., py3.12_graph_separator)
conda activate py3.12_graph_separator

# Run with defaults (uses logistics domain)
python main.py

# Run with custom domain/problem
python main.py <domain.pddl> <problem.pddl> <action_name> [max_states]
```

## Examples

```bash
# Generate DRIVE-TRUCK transitions from logistics
python main.py ../../mimir/data/logistics/domain.pddl ../../mimir/data/logistics/test_problem.pddl DRIVE-TRUCK

# Generate pick transitions from gripper domain
python main.py ../../mimir/data/gripper/domain.pddl ../../mimir/data/gripper/test_problem.pddl pick 5000
```

## How it works

1. Loads the PDDL domain and problem using Mimir's advanced API
2. Creates a lifted applicable action generator (KPKC algorithm)
3. Performs BFS from the initial state, exploring up to `max_states`
4. Filters transitions to only those matching the target action schema
5. Returns list of `(state, action, successor_state)` tuples

## Output

The script prints:
- Configuration (domain, problem, action name, max states)
- Number of matching transitions found
- First 5 transitions with state atom counts
