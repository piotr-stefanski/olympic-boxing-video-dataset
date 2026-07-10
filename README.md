# olympic-boxing-video-dataset

Repository for preparing an Olympic boxing video dataset and running object detection benchmarks. Training is done on the [Athena HPC cluster](https://docs.cyfronet.pl/spaces/~plgpawlik/pages/126648338/Athena) (ACK Cyfronet, Nvidia A100 GPUs).

## Repository structure

```
data-preparation/   – scripts for converting the raw video database into training-ready formats
experiments/        – training scripts and SLURM job definitions for each framework
  detectron2/       – Faster R-CNN, Cascade R-CNN, RetinaNet (via Detectron2)
  ultralytics/      – YOLOv8m, YOLO11m, RT-DETR-l (via Ultralytics)
docs/               – detailed HPC commands, experiment design, environment notes
```

---

## Prerequisites

1. Install `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/

2. Clone Detectron2 source **next to this repo** :
   ```
   ../detectron2/   ← Detectron2 source
   ../olympic-boxing-video-dataset/   ← this repo
   ```
   ```bash
   git clone https://github.com/facebookresearch/detectron2.git ../detectron2
   ```

3. Download the raw database (also placed next to the repo, i.e. `../datasets/olympic-boxing-video-dataset`):
   ```bash
   uv run gdown --folder 1s7KxrnJg_1CigQVIIghQ2T3Hty5NIufm -O ../datasets/olympic-boxing-video-dataset
   ```
   If you store the database elsewhere, set the `DATABASE_PATH` environment variable to that path.

4. Install dependencies (run from repo root, requires a GPU node on HPC):
   ```bash
   sbatch install_environment.sbatch
   ```
   Builds Detectron2 from `../detectron2/` and installs all other packages via `uv`.

---

## data-preparation/

Contains everything needed to convert the raw video database into formats suitable for training.

| Script | Purpose |
|---|---|
| `convert_database_to_coco_format.py` | Extracts annotated frames from video and saves COCO JSON annotations (one file per fold) |
| `convert_coco_to_yolo.py` | Converts the COCO annotations to YOLO txt format (required before Ultralytics training) |
| `config.py` | Shared constants: fold frame ranges, dataset paths |
| `database_review.py` | Sanity-check script to inspect the raw database |
| `dataset_statistics.py` | Prints per-fold image/annotation counts |

### Step 1 — Convert raw database to COCO format

```bash
uv run --frozen python data-preparation/convert_database_to_coco_format.py --save-images
```

> ⚠️ `--save-images` extracts all frames from video and saves them to disk. Requires ~156 GB of free space and takes significant time.

Output: `../datasets/olympic-boxing-video-dataset/coco_images/` and `annotations/annotations_fold_1..5.json`

### Step 2 — Convert COCO annotations to YOLO format

Required only before running Ultralytics experiments. Do this once after step 1:

```bash
uv run --frozen python data-preparation/convert_coco_to_yolo.py
```

Output: `../datasets/olympic-boxing-video-dataset/yolo_dataset/`

---

## experiments/

Contains training scripts and SLURM job definitions. All jobs are submitted from the **repo root** directory.

Experiment design: 7 cross-validation scenarios × 3 models = **21 runs per framework** (200 epochs max, patience 20, constant LR). See `docs/running_experiments.md` for the full scenario table and smoke-test instructions.

### Detectron2 (Faster R-CNN, Cascade R-CNN, RetinaNet)

```bash
# Full 21-run array (max 4 concurrent)
sbatch --array=0-20%4 experiments/detectron2/train_array.sbatch

# Collect results after training
python experiments/detectron2/collect_results.py
# Output: results.csv
```

Experiment definitions (run name, folds, model config) are in `experiments/detectron2/experiments.txt`.

### Ultralytics (YOLOv8m, YOLO11m, RT-DETR-l)

> ⚠️ **RT-DETR requires a one-time venv patch** before training. Without it, training crashes on the first batch with a PyTorch autograd error. See `docs/rtdetr_venv_patch.md` for the fix and the `sed` one-liner to re-apply it after venv recreation.

```bash
# Full 21-run array (max 4 concurrent)
sbatch --array=0-20%4 experiments/ultralytics/train_array.sbatch

# Collect results after training
python experiments/ultralytics/collect_results.py
# Output: results_ultralytics.csv
```

Experiment definitions are in `experiments/ultralytics/experiments.txt`.

### Monitoring

```bash
# Live job queue
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.9l %.6D %R"

# TensorBoard (all runs combined)
uv run tensorboard --logdir runs --port 6010 --bind_all
# Tunnel locally: ssh -L 6010:localhost:6010 <USERNAME>@athena.cyfronet.pl
```

---

## docs/

- `running_experiments.md` — smoke tests, per-framework commands, array throttle control
- `useful_commands.md` — SLURM job management, interactive GPU sessions, environment sync
- `rtdetr_venv_patch.md` — workaround for RT-DETR venv compatibility issue
