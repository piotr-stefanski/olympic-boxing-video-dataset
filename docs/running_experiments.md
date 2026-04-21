# Running experiments

## Experiment design

Both frameworks follow the same protocol:
- **Constant LR** at each model's default (no decay)
- **200 epochs** maximum, **patience 20** for early stopping
- **7 cross-validation scenarios** × 3 models = 21 runs per framework
- Best-epoch metrics reported (mAP@0.5, mAP@0.5:0.95, mAR@100)

| Scenario | Train folds | Val folds |
|----------|-------------|-----------|
| Q1 S1    | 1           | 5         |
| Q1 S2    | 1,2         | 5         |
| Q1 S3    | 1,2,3       | 5         |
| Q1 S4    | 1,2,3,4     | 5         |
| Q2 S1    | 1           | 2,3,4,5   |
| Q2 S2    | 1,2         | 3,4,5     |
| Q2 S3    | 1,2,3       | 4,5       |

---

## Detectron2 (Faster R-CNN, Cascade R-CNN, RetinaNet)

### Smoke test — interactive GPU session

```bash
srun --partition=plgrid-gpu-a100 --nodes=1 --ntasks=1 \
     --cpus-per-task=12 --gpus=1 --time=1:00:00 --pty bash

module load CUDA/12.1.1
export UV_CACHE_DIR=/net/tscratch/people/$USER/.cache/uv
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset

# Faster R-CNN
uv run --frozen python detectron2/train_detectron2.py \
    --train-folds 1 --val-folds 5 --epochs 3 --patience 2 \
    --workers 4 --output-dir runs/smoke_faster_rcnn \
    --model-config COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

# Cascade R-CNN
uv run --frozen python detectron2/train_detectron2.py \
    --train-folds 1 --val-folds 5 --epochs 3 --patience 2 \
    --workers 4 --output-dir runs/smoke_cascade_rcnn \
    --model-config Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml

# RetinaNet
uv run --frozen python detectron2/train_detectron2.py \
    --train-folds 1 --val-folds 5 --epochs 3 --patience 2 \
    --workers 4 --output-dir runs/smoke_retinanet \
    --model-config COCO-Detection/retinanet_R_50_FPN_3x.yaml
```

### Smoke test — via SLURM (single task)

```bash
# Runs row 0 only (q1_s1_faster_rcnn), confirms log redirection + GPU monitoring
sbatch --array=0-0%1 scripts/detectron2/train_array.sbatch
```

### Full 21-run array

```bash
sbatch --array=0-20%4 scripts/detectron2/train_array.sbatch
```

`%4` = max 4 concurrent jobs. Adjust with:
```bash
scontrol update JobId=<array_jobid> ArrayTaskThrottle=8
```

### Collect results

```bash
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset
python scripts/detectron2/collect_results.py
# Output: results.csv
```

---

## Ultralytics (YOLOv8m, YOLO11m, RT-DETR-l)

### One-time YOLO dataset conversion

Must be done once before any training. Converts all 5 COCO folds to YOLO format:

```bash
uv run --frozen python ultralytics/convert_coco_to_yolo.py
# Output: datasets/olympic-boxing-video-dataset/yolo_dataset/images/fold_1..5/
#         datasets/olympic-boxing-video-dataset/yolo_dataset/labels/fold_1..5/
```

### Smoke test — interactive GPU session

```bash
srun --partition=plgrid-gpu-a100 --nodes=1 --ntasks=1 \
     --cpus-per-task=12 --gpus=1 --time=1:00:00 --pty bash

module load CUDA/12.1.1
export UV_CACHE_DIR=/net/tscratch/people/$USER/.cache/uv
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset

uv run --frozen python ultralytics/train_ultralytics.py \
    --train-folds 1 --val-folds 5 \
    --epochs 3 --patience 2 --workers 4 \
    --output-dir runs/smoke_yolov8m \
    --model yolov8m.pt
```

### Smoke test — via SLURM (3 models, 1 scenario each)

Edit `scripts/ultralytics/experiments.txt` to keep only the first row of each model block,
set `--epochs 5 --patience 1` in `train_array.sbatch`, then:

```bash
sbatch --array=0-2%3 scripts/ultralytics/train_array.sbatch
```

Restore epochs/patience afterwards.

### Full 21-run array

```bash
sbatch --array=0-20%4 scripts/ultralytics/train_array.sbatch
```

### Collect results

```bash
cd /net/tscratch/people/$USER/olympic-boxing-video-dataset
python scripts/ultralytics/collect_results.py
# Output: results_ultralytics.csv
```

---

## Monitoring

```bash
# Live queue
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.9l %.6D %R"

# TensorBoard (all runs combined)
uv run tensorboard --logdir runs --port 6010 --bind_all
# Tunnel locally: ssh -L 6010:localhost:6010 <USERNAME>@athena.cyfronet.pl
```
