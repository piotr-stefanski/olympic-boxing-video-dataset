#!/bin/bash
#SBATCH --job-name=install-deps
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=install_%j.log
#SBATCH --error=install_%j.log

module load CUDA/12.1.1

cd $SLURM_SUBMIT_DIR

# Redirect uv cache to scratch (home dir has limited quota)
export UV_CACHE_DIR=/net/tscratch/people/$USER/.cache/uv

# Create venv
uv venv --python 3.11

# Install torch with CUDA 12.1 wheels first (detectron2 needs torch at build time)
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# setuptools must be present before detectron2 build (no-build-isolation uses the venv)
uv pip install "setuptools<81"

# PUT DETECTRON IN THE SAME DIRECTORY AS THE CODEBASE DIRECTORY
# Install detectron2 from pre-cloned local source (GPU nodes have no internet)
# --no-build-isolation lets it see the already-installed torch
uv pip install ../detectron2 --no-build-isolation

# Install remaining dependencies from PyPI
uv pip install \
    matplotlib \
    opencv-python-headless \
    pycocotools \
    "setuptools<81" \
    tensorboard \
    "ultralytics>=8.4.40" \
    gdown

# ==========================================
# VERIFICATION STEP
# ==========================================
echo "Installation complete. Verifying environment..."

# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Run Python inline
python - <<EOF
import torch
import detectron2
import ultralytics
print('Torch GPU:', torch.cuda.is_available())
print('All imports successful!')
EOF