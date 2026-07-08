#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=01:00:00
#SBATCH --array=0-17          # One task per 1D-ARC benchmark

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

TASK_NAME="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH

python tasks/learn_task.py --task 1DARC --name "$TASK_NAME" --model NDLM --dump-dir /work/rleap1/jan.dornhege/NDLM/outputs/NDLM_1D_ARC/c2/$TASK_NAME --config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c2.json
