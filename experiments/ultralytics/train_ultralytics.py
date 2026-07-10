"""
Ultralytics training script for the Olympic Boxing Video Dataset.

Supports YOLOv8m, YOLO11m, and RT-DETR-l. Uses model defaults for LR,
batch size, and optimizer. Generates a per-run dataset.yaml from fold
arguments, so the same yolo_dataset/ can serve all 7 cross-validation
scenarios without data duplication.

Usage:
    uv run --frozen python experiments/ultralytics/train_ultralytics.py --model yolov8m.pt
    uv run --frozen python experiments/ultralytics/train_ultralytics.py \\
        --train-folds 1 2 3 4 --val-folds 5 --model yolo11m.pt
"""
import argparse
import json
import os
import sys

import yaml
from ultralytics import YOLO

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, '../../data-preparation')))
from convert_coco_to_yolo import convert_fold, DATASET_DIR as _DEFAULT_DATASET_DIR

_IOU_LABELS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

CATEGORIES = [
    'Punch to the head with the left hand',
    'Punch to the head with the right hand',
    'Punch to the torso with the left hand',
    'Punch to the torso with the right hand',
    'Block with the left hand',
    'Block with the right hand',
    'Missed punch with the left hand',
    'Missed punch with the right hand',
]


def ensure_folds(folds: list[int], dataset_dir: str):
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    images_dir = os.path.join(dataset_dir, 'coco_images')
    yolo_dir = os.path.join(dataset_dir, 'yolo_dataset')
    for fold in folds:
        fold_dir = os.path.join(yolo_dir, 'images', f'fold_{fold}')
        if not os.path.isdir(fold_dir):
            print(f'Converting fold {fold} to YOLO format...')
            count = convert_fold(fold, annotations_dir, images_dir, yolo_dir)
            print(f'  fold_{fold}: {count} images')


def write_dataset_yaml(output_dir: str, yolo_dir: str,
                       train_folds: list[int], val_folds: list[int]) -> str:
    data = {
        'path': os.path.abspath(yolo_dir),
        'train': [f'images/fold_{f}' for f in train_folds],
        'val': [f'images/fold_{f}' for f in val_folds],
        'nc': len(CATEGORIES),
        'names': CATEGORIES,
    }
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return yaml_path


def add_per_iou_callback(model):
    """
    Register a callback that logs per-IoU mAP (map50..map95) to TensorBoard
    after each validation epoch, matching the Detectron2 DetailedCOCOEvaluator.
    box.all_ap is [n_classes, 10]; mean over classes gives per-IoU mAP.
    """
    def on_fit_epoch_end(trainer):
        try:
            from ultralytics.utils.callbacks.tensorboard import WRITER
            if WRITER is None:
                return
            all_ap = trainer.validator.metrics.box.all_ap  # [n_classes, 10], values in [0, 1]
            per_iou = all_ap.mean(0)  # [10]
            step = trainer.epoch + 1
            for iou_label, val in zip(_IOU_LABELS, per_iou):
                WRITER.add_scalar(f'bbox/map{iou_label}', float(val) * 100, step)
        except Exception:
            pass

    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)


def extract_per_iou_metrics(metrics) -> dict:
    """Extract per-IoU mAP from a val metrics object, scaled to 0-100."""
    all_ap = metrics.box.all_ap  # [n_classes, 10], values in [0, 1]
    per_iou = all_ap.mean(0)    # [10], mean over classes
    result = {f'map{iou}': float(v) * 100 for iou, v in zip(_IOU_LABELS, per_iou)}
    result['map50_95'] = float(metrics.box.map) * 100
    result['map50'] = float(metrics.box.map50) * 100
    result['map75'] = float(metrics.box.map75) * 100
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folds', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--val-folds', nargs='+', type=int, default=[5])
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Ultralytics weight file (e.g. yolov8m.pt, yolo11m.pt, rtdetr-l.pt)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience in epochs (0 = disabled)')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='output/ultralytics_run')
    parser.add_argument('--dataset-dir', type=str, default=_DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    print(f'Train folds: {args.train_folds}')
    print(f'Val folds:   {args.val_folds}')
    print(f'Model:       {args.model}')

    all_folds = sorted(set(args.train_folds + args.val_folds))
    ensure_folds(all_folds, args.dataset_dir)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    yolo_dir = os.path.join(args.dataset_dir, 'yolo_dataset')
    yaml_path = write_dataset_yaml(output_dir, yolo_dir, args.train_folds, args.val_folds)
    print(f'Dataset yaml: {yaml_path}')

    model = YOLO(args.model)
    add_per_iou_callback(model)

    model.train(
        data=yaml_path,
        epochs=args.epochs,
        patience=args.patience if args.patience > 0 else 0,
        workers=args.workers,
        device=0,
        project=os.path.dirname(output_dir),
        name=os.path.basename(output_dir),
        exist_ok=True,
        save_period=10,
        pretrained=True,
        verbose=True,
        lrf=1.0,  # constant LR: final LR = lr0 * lrf = lr0 (no decay, matching Detectron2 setup)
    )

    print('\n--- Final Validation (best.pt) ---')
    metrics = model.val(data=yaml_path, device=0, workers=args.workers)
    per_iou = extract_per_iou_metrics(metrics)

    print('\n=== Per-IoU metrics (scaled 0-100) ===')
    for k, v in per_iou.items():
        print(f'  {k}: {v:.3f}')

    with open(os.path.join(output_dir, 'per_iou_metrics.json'), 'w') as f:
        json.dump(per_iou, f, indent=2)
    print(f'Saved per-IoU metrics to {output_dir}/per_iou_metrics.json')


if __name__ == '__main__':
    main()
