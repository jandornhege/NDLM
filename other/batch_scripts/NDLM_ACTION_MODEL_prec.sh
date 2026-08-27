#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=01:00:00
# SBATCH --array=0-0          # One task per 1D-ARC benchmark

TASK_NAMES=(
	"navigation-xy"
)

TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

python tasks/learn_task.py \
	--task ActionModel \
	--domain "$TASK_NAME" \
	--model NDLM \
	--dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NDLM_ACTION_MODEL_prec/L2/$TASK_NAME \
	--data-path /work/rleap1/jan.dornhege/NDLM/src/data/data_more_expressive_gp/ \
	--remove-arguments \
    --precondition \
	--test-interval 10 \
	--num-train-states 30 \
	--num-test-states 30 \
	--num-epochs 5000 \
	--learning-rate 0.001 \
	--weight-decay 0.0001 \
	--hidden-roles 4 \
	--hidden-concepts 4 \
	--num-layers 2 \
	--mode strict \
	--num-epochs 5000 \
	--test-interval 10 \
	--batch-size 4 \
	--activation-function identity \

	# --config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c2.json \
	# --test-interval 10 \
	# --num-train-states 30 \
	# --num-test-states 30 \
	# --num-epochs 5000 \
	# --learning-rate 0.001 \
	# --weight-decay 0.0001
