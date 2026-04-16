#!/bin/bash
#SBATCH --job-name=train-rcnn
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --time=15:00:00
#SBATCH --output=output/train_%j.log
#SBATCH --error=output/train_%j.log

export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR

# Monitor GPU utilization (every 5s) — check output/gpu_usage_*.csv after job
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 5 > output/gpu_usage_${SLURM_JOB_ID}.csv 2>&1 &
GPU_MON_PID=$!

# Monitor CPU/RAM utilization (every 10s) — check output/cpu_usage_*.log after job
(while true; do
  echo "$(date +%T) $(ps -u $USER -o %cpu,%mem --no-headers | awk '{c+=$1; m+=$2} END {print "CPU:"c"% MEM:"m"%"}')"
  sleep 10
done) > output/cpu_usage_${SLURM_JOB_ID}.log 2>&1 &
CPU_MON_PID=$!

uv run main.py

# Cleanup monitoring processes
kill $GPU_MON_PID $CPU_MON_PID 2>/dev/null
