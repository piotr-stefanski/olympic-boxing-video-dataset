#!/bin/bash
#SBATCH --job-name=convert-db
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=output/convert_database_%j.log
#SBATCH --error=output/convert_database_%j.log

export PYTHONUNBUFFERED=1

cd $SLURM_SUBMIT_DIR

uv run convert_database_to_coco_format.py --save-images
