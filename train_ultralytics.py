"""
Training script for Ultralytics-based object detectors on the boxing dataset.

Supports: YOLOv8, YOLOv12, RT-DETR, YOLO26.
Automatically merges COCO annotation folds (1-4) if not already merged.

Usage:
    uv run train_ultralytics.py --model yolov8m
    uv run train_ultralytics.py --model yolov12m --epochs 100 --batch 32
    uv run train_ultralytics.py --model rtdetr-l --imgsz 640
"""
import argparse
import os

from ultralytics import YOLO

from convert_coco_to_yolo import prepare_yolo_dataset

# Mapping of CLI model names to Ultralytics pretrained weight files
MODEL_REGISTRY = {
    # YOLOv8 variants
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8x': 'yolov8x.pt',
    # YOLOv12 variants
    'yolov12n': 'yolov12n.pt',
    'yolov12s': 'yolov12s.pt',
    'yolov12m': 'yolov12m.pt',
    'yolov12l': 'yolov12l.pt',
    'yolov12x': 'yolov12x.pt',
    # RT-DETR variants
    'rtdetr-l': 'rtdetr-l.pt',
    'rtdetr-x': 'rtdetr-x.pt',
    # YOLO26 variants
    'yolo26n': 'yolo26n.pt',
    'yolo26s': 'yolo26s.pt',
    'yolo26m': 'yolo26m.pt',
    'yolo26l': 'yolo26l.pt',
    'yolo26x': 'yolo26x.pt',
}

DATASET_DIR = '../datasets/olympic-boxing-video-dataset'
TRAIN_FOLDS = [1, 2, 3, 4]
VAL_FOLD = 5


def ensure_yolo_dataset():
    """Convert COCO annotations to YOLO format if not already done."""
    yolo_dir = os.path.join(DATASET_DIR, 'yolo_dataset')
    if os.path.exists(yolo_dir):
        print(f'YOLO dataset already exists: {yolo_dir}')
        return

    print('YOLO dataset not found. Converting from COCO format...')
    prepare_yolo_dataset(DATASET_DIR, TRAIN_FOLDS, VAL_FOLD)


def main():
    parser = argparse.ArgumentParser(
        description='Train Ultralytics object detectors on the boxing dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f'Model to train. Choices: {", ".join(MODEL_REGISTRY.keys())}'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--lr0', type=float, default=0.005, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer (default: SGD)')
    parser.add_argument('--device', type=str, default='0', help='Device to train on (default: 0)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')

    args = parser.parse_args()

    # Ensure YOLO format dataset exists
    ensure_yolo_dataset()

    # Load pretrained model
    weights = MODEL_REGISTRY[args.model]
    print(f'\nLoading model: {args.model} (weights: {weights})')
    model = YOLO(weights)

    # Dataset config path (relative to this script)
    dataset_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.yaml')

    # Train
    print(f'\nStarting training: {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}')
    print(f'Optimizer: {args.optimizer}, lr0={args.lr0}')
    print(f'Dataset config: {dataset_yaml}\n')

    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        momentum=0.9,
        weight_decay=0.0005,
        device=args.device,
        workers=args.workers,
        project='output',
        name=args.model,
        save_period=10,          # Save checkpoint every 10 epochs (matching Faster R-CNN setup)
        exist_ok=True,           # Allow reusing the same output directory
        pretrained=True,
        resume=args.resume,
        verbose=True,
    )

    # Validate after training
    print('\n--- Final Validation ---')
    metrics = model.val(data=dataset_yaml, device=args.device, workers=args.workers)
    print(f'mAP@0.5:0.95 = {metrics.box.map:.4f}')
    print(f'mAP@0.5      = {metrics.box.map50:.4f}')
    print(f'mAP@0.75     = {metrics.box.map75:.4f}')
    print(f'mAR@100      = {metrics.box.mr:.4f}')

    print(f'\nTraining complete. Results saved to output/{args.model}/')


if __name__ == '__main__':
    main()
