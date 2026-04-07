"""
Detectron2 training script for the Olympic Boxing Video Dataset.

Trains a Faster R-CNN R50-FPN on configurable train/val fold combinations.

Usage:
    uv run train_detectron2.py
    uv run train_detectron2.py --train-folds 1 2 3 4 --val-folds 5
    uv run train_detectron2.py --train-folds 1 2 --val-folds 3 4 5 --epochs 200
"""
import argparse
import os
import math

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


DATASET_DIR = '../datasets/olympic-boxing-video-dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'coco_images')
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
NUM_CLASSES = 8  # boxing action categories (see data-utils/config.py)


def register_folds(fold_numbers: list[int], split_name: str) -> list[str]:
    """Register each fold as a separate Detectron2 dataset and return their names."""
    registered = []
    for fold in fold_numbers:
        name = f'boxing_{split_name}_fold_{fold}'
        ann_path = os.path.abspath(os.path.join(ANNOTATIONS_DIR, f'annotations_fold_{fold}.json'))
        img_path = os.path.abspath(IMAGES_DIR)

        if not os.path.exists(ann_path):
            raise FileNotFoundError(f'Annotation file not found: {ann_path}')

        # Avoid double-registration if script re-runs in same process
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        register_coco_instances(name, {}, ann_path, img_path)
        registered.append(name)
        print(f'  Registered {name}: {ann_path}')
    return registered


def count_train_images(train_dataset_names: list[str]) -> int:
    total = 0
    for name in train_dataset_names:
        total += len(DatasetCatalog.get(name))
    return total


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval')
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def build_cfg(train_datasets, val_datasets, output_dir, epochs, ims_per_batch, base_lr, num_workers):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')

    cfg.DATASETS.TRAIN = tuple(train_datasets)
    cfg.DATASETS.TEST = tuple(val_datasets)

    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr

    # Convert epochs -> iterations
    num_train_imgs = count_train_images(train_datasets)
    iters_per_epoch = max(1, math.ceil(num_train_imgs / ims_per_batch))
    cfg.SOLVER.MAX_ITER = epochs * iters_per_epoch
    # Save checkpoint every 10 epochs (matches ultralytics setup)
    cfg.SOLVER.CHECKPOINT_PERIOD = 10 * iters_per_epoch
    # Evaluate every epoch
    cfg.TEST.EVAL_PERIOD = iters_per_epoch

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f'\nTraining images: {num_train_imgs}')
    print(f'Iterations per epoch: {iters_per_epoch}')
    print(f'Total iterations ({epochs} epochs): {cfg.SOLVER.MAX_ITER}')
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folds', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--val-folds', nargs='+', type=int, default=[5])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='output/detectron2_run')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    print(f'Train folds: {args.train_folds}')
    print(f'Val folds:   {args.val_folds}')

    print('\nRegistering datasets...')
    train_names = register_folds(args.train_folds, 'train')
    val_names = register_folds(args.val_folds, 'val')

    cfg = build_cfg(
        train_datasets=train_names,
        val_datasets=val_names,
        output_dir=args.output_dir,
        epochs=args.epochs,
        ims_per_batch=args.batch,
        base_lr=args.lr,
        num_workers=args.workers,
    )

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
