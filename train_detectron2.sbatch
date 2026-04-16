#!/bin/bash
#SBATCH --job-name=train-detectron2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=output/train_detectron2_%j.log
#SBATCH --error=output/train_detectron2_%j.log

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export UV_CACHE_DIR=/net/tscratch/people/plgkamilkot/.cache/uv

module load CUDA/12.1.1

cd $SLURM_SUBMIT_DIR
mkdir -p output

# Monitor GPU utilization (every 5s)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 5 > output/gpu_usage_detectron2_${SLURM_JOB_ID}.csv 2>&1 &
GPU_MON_PID=$!

# Monitor CPU/RAM utilization (every 10s)
(while true; do
  echo "$(date +%T) $(ps -u $USER -o %cpu,%mem --no-headers | awk '{c+=$1; m+=$2} END {print "CPU:"c"% MEM:"m"%"}')"
  sleep 10
done) > output/cpu_usage_detectron2_${SLURM_JOB_ID}.log 2>&1 &
CPU_MON_PID=$!

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

uv run --frozen python train_detectron2.py \
    --train-folds 1 2 3 4 \
    --val-folds 5 \
    --epochs 4 \
    --batch 16 \
    --lr 0.01 \
    --workers 10 \
    --output-dir output/detectron2_baseline

# Cleanup monitoring processes
kill $GPU_MON_PID $CPU_MON_PID 2>/dev/null
