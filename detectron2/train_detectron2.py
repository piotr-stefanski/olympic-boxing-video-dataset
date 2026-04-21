"""
Detectron2 training script for the Olympic Boxing Video Dataset.

Uses the provider's default config (LR, batch size, warmup) unchanged.
Only sets NUM_CLASSES, DATASETS, OUTPUT_DIR, MAX_ITER, EVAL_PERIOD,
CHECKPOINT_PERIOD, and disables SOLVER.STEPS for a constant LR schedule.

Usage:
    uv run train_detectron2.py
    uv run train_detectron2.py --train-folds 1 2 3 4 --val-folds 5
    uv run train_detectron2.py --model-config COCO-Detection/retinanet_R_50_FPN_3x.yaml
"""
import argparse
import os
import math

import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import get_event_storage


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '../../datasets/olympic-boxing-video-dataset'))
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


class DetailedCOCOEvaluator(COCOEvaluator):
    """
    Extended COCOEvaluator that extracts per-IoU mAP (map50..map95),
    mAP@0.5:0.95, and mAR@100, pushing them to EventStorage (TensorBoard).
    """
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        if iou_type != 'bbox' or coco_eval is None:
            return results

        try:
            precision = coco_eval.eval['precision']  # [T, R, K, A, M]
        except Exception:
            return results

        iou_thresholds = np.linspace(0.5, 0.95, 10)
        per_iou_ap = {}
        for t, iou in enumerate(iou_thresholds):
            p = precision[t, :, :, 0, -1]  # [R, K]
            valid = p[p > -1]
            ap = float(np.mean(valid) * 100) if valid.size else float('nan')
            per_iou_ap[f'map{int(round(iou * 100))}'] = ap

        mean_ap = float(np.nanmean(list(per_iou_ap.values())))
        mar_100 = float(coco_eval.stats[8] * 100) if coco_eval.stats is not None else float('nan')

        try:
            storage = get_event_storage()
            for k, v in per_iou_ap.items():
                storage.put_scalar(f'bbox/{k}', v, smoothing_hint=False)
            storage.put_scalar('bbox/map50_95', mean_ap, smoothing_hint=False)
            storage.put_scalar('bbox/mar100', mar_100, smoothing_hint=False)
        except AssertionError:
            pass

        results.update(per_iou_ap)
        results['map50_95'] = mean_ap
        results['mar100'] = mar_100

        print('\n=== Custom metrics (scaled to 0-100) ===')
        for k, v in per_iou_ap.items():
            print(f'  {k}: {v:.3f}')
        print(f'  map50_95: {mean_ap:.3f}')
        print(f'  mar100:   {mar_100:.3f}')

        return results


class _EarlyStopSignal(BaseException):
    """Inherits BaseException so Detectron2's `except Exception` logger is bypassed,
    but `finally: after_train()` still runs to save model_final.pth."""
    pass


class EarlyStoppingHook(HookBase):
    """Stop training when bbox/map50_95 has not improved for `patience` evals."""
    def __init__(self, eval_period: int, patience: int, metric: str = 'bbox/map50_95'):
        self._eval_period = eval_period
        self._patience = patience
        self._metric = metric
        self._best = float('-inf')
        self._no_improve = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._eval_period != 0:
            return

        try:
            val = self.trainer.storage.history(self._metric).latest()
        except (KeyError, AttributeError):
            return

        if val > self._best:
            self._best = val
            self._no_improve = 0
        else:
            self._no_improve += 1
            epoch = next_iter // self._eval_period
            print(f'[EarlyStopping] No improvement for {self._no_improve}/{self._patience} '
                  f'epochs (best {self._metric}: {self._best:.3f})')
            if self._no_improve >= self._patience:
                print(f'[EarlyStopping] Patience exhausted at epoch {epoch}. Stopping.')
                raise _EarlyStopSignal()


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval')
        os.makedirs(output_folder, exist_ok=True)
        return DetailedCOCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        best_hook = BestCheckpointer(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric='bbox/map50_95',
            mode='max',
            file_prefix='model_best',
        )
        hooks.insert(-1, best_hook)
        return hooks


def is_retinanet(model_config: str) -> bool:
    return 'retinanet' in model_config.lower()


def is_cascade(model_config: str) -> bool:
    return 'cascade' in model_config.lower()


def build_cfg(train_datasets, val_datasets, output_dir, epochs, num_workers, model_config):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

    cfg.DATASETS.TRAIN = tuple(train_datasets)
    cfg.DATASETS.TEST = tuple(val_datasets)

    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Use default IMS_PER_BATCH from provider config to compute epoch length.
    ims_per_batch = cfg.SOLVER.IMS_PER_BATCH
    num_train_imgs = count_train_images(train_datasets)
    iters_per_epoch = max(1, math.ceil(num_train_imgs / ims_per_batch))

    cfg.SOLVER.MAX_ITER = epochs * iters_per_epoch
    # Disable LR decay steps (constant LR as per experiment design).
    cfg.SOLVER.STEPS = ()
    # Rolling checkpoint every 10 epochs, keep only the latest 1.
    cfg.SOLVER.CHECKPOINT_PERIOD = 10 * iters_per_epoch
    cfg.SOLVER.MAX_TO_KEEP = 1
    # Evaluate every epoch.
    cfg.TEST.EVAL_PERIOD = iters_per_epoch

    if is_retinanet(model_config):
        cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    # Cascade Mask R-CNN zoo config has a mask head; disable it since our
    # dataset has no segmentation annotations.
    if is_cascade(model_config):
        cfg.MODEL.MASK_ON = False

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    print(f'\nModel:               {model_config}')
    print(f'Default LR:          {cfg.SOLVER.BASE_LR}')
    print(f'Default batch size:  {ims_per_batch}')
    print(f'Training images:     {num_train_imgs}')
    print(f'Iterations/epoch:    {iters_per_epoch}')
    print(f'Total iterations:    {cfg.SOLVER.MAX_ITER}  ({epochs} epochs)')
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folds', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--val-folds', nargs='+', type=int, default=[5])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience in epochs (0 = disabled)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='output/detectron2_run')
    parser.add_argument('--model-config', type=str,
                        default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='Detectron2 model zoo config path')
    parser.add_argument('--no-resume', action='store_true',
                        help='Disable auto-resume from existing checkpoints in output-dir')
    args = parser.parse_args()
    resume = not args.no_resume

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
        num_workers=args.workers,
        model_config=args.model_config,
    )

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    if args.patience > 0:
        trainer.register_hooks([
            EarlyStoppingHook(cfg.TEST.EVAL_PERIOD, args.patience)
        ])

    try:
        trainer.train()
    except _EarlyStopSignal:
        pass


if __name__ == '__main__':
    main()
