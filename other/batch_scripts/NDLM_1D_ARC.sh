#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=01:00:00
#SBATCH --array=0-359         # 18 tasks x 2 configurations x 10 runs
#SBATCH --qos=rleap_deadline

TASK_NAMES=(
	1d_mirror
	1d_move_2p
	1d_recolor_cmp
	1d_hollow
	1d_denoising_1c
	1d_denoising_mc
	1d_move_3p
	1d_scale_dp
	1d_flip
	1d_padded_fill
	1d_pcopy_1c
	1d_recolor_cnt
	1d_fill
	1d_move_dp
	1d_move_2p_dp
	1d_pcopy_mc
	1d_move_1p
	1d_recolor_oe
)

CONFIG_IDS=(0 2)
NUM_TASKS=${#TASK_NAMES[@]}
NUM_CONFIGS=${#CONFIG_IDS[@]}

RUN_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_CONFIGS * NUM_TASKS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_CONFIGS * NUM_TASKS)))
CONFIG_INDEX=$((REMAINDER / NUM_TASKS))
TASK_INDEX=$((REMAINDER % NUM_TASKS))

TASK_NAME="${TASK_NAMES[$TASK_INDEX]}"
CONFIG_ID="${CONFIG_IDS[$CONFIG_INDEX]}"
CONFIG_FILE="/work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c${CONFIG_ID}.json"
OUTPUT_DIR="/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_1D_ARC/c${CONFIG_ID}/run_${RUN_INDEX}/$TASK_NAME"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME (config c${CONFIG_ID}, run ${RUN_INDEX})"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

python tasks/learn_task.py --task 1DARC --arc-task-name "$TASK_NAME" --model NDLM --dump-dir "$OUTPUT_DIR" --config-file "$CONFIG_FILE"
