#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=06:00:00
#SBATCH --array=0-0         # One task per 1D-ARC benchmark

TASK_NAMES=(
        moose_ferry
        # clear
        # miconic
        # delivery
        # npuzzle
        # logistics
        # gripper_single_goal
        # gripper
)

TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

python tasks/learn_task.py \
	--task optGP \
	--name "$TASK_NAME" \
	--model NDLM \
	--dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/c2/$TASK_NAME \
	--config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c2.json \
        --max-sampling-seconds-per-problem 60
