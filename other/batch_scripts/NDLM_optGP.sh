#!/bin/bash
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --mem=64g                   # Memory allocation
#SBATCH --partition=rleap_gpu_24gb  # Partition (queue) to use
#SBATCH --output=/work/rleap1/jan.dornhege/B_Runs/%A_%a.txt  # Output log file per array task
#SBATCH --time=12:00:00
#SBATCH --qos=rleap_deadline
#SBATCH --array=0-49       # Ten runs for each configured optGP domain

TASK_NAMES=(
        blocks
        gripper
        logistics
        miconic
        navigation-xy
)

RUN_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TASK_NAMES[@]}))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TASK_NAMES[@]}))
TASK_NAME="${TASK_NAMES[$TASK_INDEX]}"
RUN_ID="run_${RUN_INDEX}"
declare -A CONFIG_FILES=(
        [blocks]="blocks_optGP.json"
        [gripper]="gripper_optGP.json"
        [logistics]="logistics_optGP.json"
        [miconic]="miconig_optGP.json"
        [navigation-xy]="navigation-xy_optGP.json"
)
CONFIG_FILE="/work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/${CONFIG_FILES[$TASK_NAME]}"
RUN_ROOT="/work/rleap1/jan.dornhege/NDLM/outputs/NDLM_OPTGP_300sec_300states"
TRAIN_ROOT="$RUN_ROOT/sandbox/$TASK_NAME/$RUN_ID"
TEST_ROOT="$RUN_ROOT/test_only/$TASK_NAME/$RUN_ID"
mkdir -p "$TRAIN_ROOT" "$TEST_ROOT"

# Activate virtual environment
source /work/rleap1/jan.dornhege/envs/py3.12_graph_separator/bin/activate

echo "Activated venv, starting task: $TASK_NAME (run ${RUN_INDEX})"
echo "Using config: $CONFIG_FILE"

cd /work/rleap1/jan.dornhege/NDLM/src/

export PYTHONPATH=/work/rleap1/jan.dornhege/neural-logic-machines:/work/rleap1/jan.dornhege/neural-logic-machines/third_party/Jacinle:$PYTHONPATH


python tasks/learn_task.py \
        --task optGP \
        --domain "$TASK_NAME" \
        --num-train-states 300 \
        --num-test-states 0 \
        --sampling-method opt_then_bfs \
	--model NDLM \
        --test-interval 10 \
        --dump-dir "$TRAIN_ROOT" \
        --data-path /work/rleap1/jan.dornhege/NDLM/src/data/data_more_expressive_gp/ \
        --config-file "$CONFIG_FILE" \
        --max-sampling-seconds-per-problem 300 \
        --remove-arguments \
        --test_supervised \
        --loss-type BCE \
        --weighted_loss \
        --learning-rate 0.001 \
        --weight-decay 0.0001

find_valid_run_dir() {
        local root="$1"
        find "$root" -mindepth 1 -maxdepth 1 -type d -print0 \
                | while IFS= read -r -d '' dir; do
                        if [[ -f "$dir/args.txt" ]]; then
                                printf '%s\n' "$dir"
                        fi
                  done \
                | sort -V \
                | tail -n 1
}

CHECKPOINT_ROOT="$(find_valid_run_dir "$TRAIN_ROOT")"
if [[ -z "$CHECKPOINT_ROOT" ]]; then
        echo "No valid optGP training experiment directory with args.txt found under $TRAIN_ROOT" >&2
        exit 1
fi

echo "Testing optGP policy from: $CHECKPOINT_ROOT"

python tasks/learn_task.py \
        --task optGP \
        --domain "$TASK_NAME" \
        --model NDLM \
        --data-path /work/rleap1/jan.dornhege/NDLM/src/data/data_more_expressive_gp/ \
        --test-only \
        --checkpoint-path "$CHECKPOINT_ROOT" \
        --dump-dir "$TEST_ROOT" \
        --num-test-states 200

        # --input-residual \
        # --transitive-closure \
	# --config-file /work/rleap1/jan.dornhege/NDLM/src/ndlm/config_files/c5.json \