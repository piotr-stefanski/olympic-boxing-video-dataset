# Useful commands

## Navigation

```bash
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset
DATABASE_PATH=/net/tscratch/people/$USER/datasets/olympic-boxing-video-dataset
```

## SLURM job management

```bash
# Formatted queue view
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.9l %.6D %R"

# Job history and efficiency
hpc-jobs-history
seff <jobid>

# Other essentials
sbatch <script>
scancel <jobid>
sacct

# Live-update array throttle without cancelling
scontrol update JobId=<array_jobid> ArrayTaskThrottle=8
```

## Interactive GPU session (smoke testing)

```bash
srun --partition=plgrid-gpu-a100 --nodes=1 --ntasks=1 \
     --cpus-per-task=12 --gpus=1 --time=1:00:00 --pty bash

# Then inside the session:
module load CUDA/12.1.1
export UV_CACHE_DIR=/net/tscratch/people/$USER/.cache/uv
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset
```

## Environment sync (before any SLURM job submission)

```bash
module load CUDA/12.1.1
uv sync --no-build-isolation
```

## TensorBoard

```bash
# On Athena (pick a free port, e.g. 6010)
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset
uv run tensorboard --logdir runs --port 6010 --bind_all

# On your local machine (SSH tunnel)
ssh -L 6010:localhost:6010 <USERNAME>@athena.cyfronet.pl
# Then open http://localhost:6010
```

## Collect results

```bash
# Detectron2
python scripts/detectron2/collect_results.py

# Ultralytics
python scripts/ultralytics/collect_results.py
```
