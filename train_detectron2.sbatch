#!/bin/bash
#SBATCH --job-name=train-detectron2
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
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

uv run --frozen python train_detectron2.py \
    --train-folds 1 2 3 4 \
    --val-folds 5 \
    --epochs 2 \
    --batch 8 \
    --output-dir output/detectron2_baseline
