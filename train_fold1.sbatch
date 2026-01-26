#!/bin/bash
#SBATCH --job-name=train_fold1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=train_fold1_%j.log
#SBATCH --error=train_fold1_%j.err

# Change to project directory
cd /net/tscratch/people/plgpiotrstefanski/projects/olympic-boxing-video-dataset

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Disable Python output buffering for real-time log streaming
export PYTHONUNBUFFERED=1

# Run training
uv run python train.py --epochs 10 --batch_size 32 --num_workers 8 --train_folds=1 --output_dir ./output

echo "Job finished at: $(date)"
