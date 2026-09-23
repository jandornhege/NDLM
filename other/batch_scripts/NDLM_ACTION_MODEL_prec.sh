#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=1          # CPU cores per task
#SBATCH --gres=shard:3          # GPUs per node
#SBATCH --mem=8g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=1-00:00:00
#SBATCH --qos=rleap_deadline
#SBATCH --array=0-29        # One task for the configured Action Model domain

TASK_NAMES=(
	clear
	npuzzle
	delivery
	logistics_4_actions_single_goal
)


TASK_NAME="${TASK_NAMES[$((SLURM_ARRAY_TASK_ID % 3))]}"
# TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"
# TASK_NAME="${TASK_NAMES[$((SLURM_ARRAY_TASK_ID % 3))]}"
# TASK_NAME="logistics_4_actions_single_goal"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

run_index=$((SLURM_ARRAY_TASK_ID//3))

# for run_index in $(seq 1 10); do
# 	echo "Starting repeated run $run_index/10 for $TASK_NAME"
python tasks/learn_task.py \
	--task ActionModel \
	--domain "$TASK_NAME" \
	--model NDLM \
	--dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NDLM_ACTION_MODEL_prec/full_app_3/${run_index}/$TASK_NAME \
	--data-path /work/rleap1/jan.dornhege/NDLM/src/data/ \
	--remove-arguments \
    --precondition \
	--full-applicability \
	--test-interval 10 \
	--num-epochs 1000 \
	--num-train-states 600 \
	--num-test-states 300 \
	--learning-rate 0.001 \
	--weight-decay 0.0001 \
	--hidden-roles 10 \
	--hidden-concepts 10 \
	--num-layers 4 \
	--mode strict \
	--batch-size 4 \
	--activation-function identity \

