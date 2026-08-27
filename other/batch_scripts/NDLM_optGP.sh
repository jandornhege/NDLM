#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=12:00:00
#SBATCH --array=0-0         # One task per 1D-ARC benchmark

TASK_NAMES=(
        # on
        # moose_ferry
        # clear
        # miconic
        # delivery
        # npuzzle
        # logistics
        # logistics_4_actions_single_goal
        # gripper_single_goal
        # gripper
        # navigation-xy
        # grid
        logistics
        blocks
        blocks-m
        grid
        miconic
        visitall
        visitall-xy
)

TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH


python tasks/learn_task.py \
        --task optGP \
        --domain "$TASK_NAME" \
        --num-train-states 300 \
        --num-test-states 0 \
        --sampling-method opt_then_bfs \
	--model NLM \
        --test-interval 10 \
        --nlm-breadth 3 \
	--dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP/sandbox/$TASK_NAME/ \
        --data-path /work/rleap1/jan.dornhege/NDLM/src/data/data_more_expressive_gp/ \
        --max-sampling-seconds-per-problem 300 \
        --remove-arguments \
        --test_supervised \
        --hidden-roles 10 \
        --hidden-concepts 10 \
        --num-layers 4 \
        --mode strict \
        --num-epochs 1000 \
        --test-interval 10 \
        --batch-size 10 \
        --activation-function identity \
        --loss-type BCE \
        --weighted_loss \
        --learning-rate 0.001 \
        --weight-decay 0.0001

        # --input-residual \
        # --transitive-closure \
	# --config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c5.json \