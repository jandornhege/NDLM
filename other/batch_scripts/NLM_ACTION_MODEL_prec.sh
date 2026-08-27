#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=01:00:00
#SBATCH --array=0-31         # 4 tasks × 8 configs (c5-c12)

TASK_NAMES=(
	clear
	npuzzle
	logistics
	delivery
)

# CONFIGS=(c5 c6 c7 c8 c9 c10 c11 c12)

CONFIGS=(c13 c14 c15 c16 c17 c18 c19 c20)

TASK_ID=$((SLURM_ARRAY_TASK_ID % 4))
CONFIG_ID=$((SLURM_ARRAY_TASK_ID / 4))

TASK_NAME="${TASK_NAMES[$TASK_ID]}"
CONFIG="${CONFIGS[$CONFIG_ID]}"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME (config: $CONFIG)"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

python tasks/learn_task.py \
	--task ActionModel \
	--name "$TASK_NAME" \
	--model NLM \
	--dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NLM_ACTION_MODEL_prec/$CONFIG/$TASK_NAME \
	--config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/$CONFIG.json \
    --precondition