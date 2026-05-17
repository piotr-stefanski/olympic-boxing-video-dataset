#!/bin/bash
#SBATCH --job-name=boxing-array
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --time=48:00:00
# Bootstrap log: only the few lines before we know OUTPUT_DIR land here.
# After the exec redirect below, all output goes to runs/<run_name>/slurm_<jobid>.log
#SBATCH --output=runs/_array_bootstrap/%x_%A_%a.log
#SBATCH --error=runs/_array_bootstrap/%x_%A_%a.log
# NOTE: set --array=0-N%K when submitting, e.g.:
#   sbatch --array=0-20%4 experiments/detectron2/train_array.sbatch
# where N = (num non-comment lines in experiments.txt) - 1  [currently 20]
# and K = max concurrent jobs (start with 4).

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export UV_CACHE_DIR=/net/tscratch/people/$USER/.cache/uv

module load CUDA/12.1.1

cd $SLURM_SUBMIT_DIR
mkdir -p runs/_array_bootstrap

EXPERIMENTS_FILE="experiments/detectron2/experiments.txt"

# Pick the (TASK_ID+1)-th non-comment, non-blank line
LINE=$(grep -v '^\s*#' "$EXPERIMENTS_FILE" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$LINE" ]; then
    echo "No experiment found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse: run_name train_folds val_folds model_config
read -r RUN_NAME TRAIN_FOLDS VAL_FOLDS MODEL_CONFIG <<< "$LINE"

OUTPUT_DIR="runs/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect everything from here on into the run dir
exec > "${OUTPUT_DIR}/slurm_${SLURM_JOB_ID}.log" 2>&1

# Convert "1,2,3" -> "1 2 3" for argparse nargs='+'
TRAIN_ARGS=$(echo "$TRAIN_FOLDS" | tr ',' ' ')
VAL_ARGS=$(echo "$VAL_FOLDS" | tr ',' ' ')

echo "=========================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run name:   $RUN_NAME"
echo "Train folds: $TRAIN_ARGS"
echo "Val folds:   $VAL_ARGS"
echo "Model:      $MODEL_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "=========================================="

# Monitor GPU utilization (every 5s)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 5 > "${OUTPUT_DIR}/gpu_usage_${SLURM_JOB_ID}.csv" 2>&1 &
GPU_MON_PID=$!

# Monitor CPU/RAM (every 10s)
(while true; do
  echo "$(date +%T) $(ps -u $USER -o %cpu,%mem --no-headers | awk '{c+=$1; m+=$2} END {print "CPU:"c"% MEM:"m"%"}')"
  sleep 10
done) > "${OUTPUT_DIR}/cpu_usage_${SLURM_JOB_ID}.log" 2>&1 &
CPU_MON_PID=$!

uv run --frozen python experiments/detectron2/train_detectron2.py \
    --train-folds $TRAIN_ARGS \
    --val-folds $VAL_ARGS \
    --epochs 200 \
    --patience 20 \
    --workers 10 \
    --output-dir "$OUTPUT_DIR" \
    --model-config "$MODEL_CONFIG"

kill $GPU_MON_PID $CPU_MON_PID 2>/dev/null