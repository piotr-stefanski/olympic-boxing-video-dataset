"""
Evaluation script for Faster R-CNN on Olympic Boxing Dataset.

Computes mAP (mean Average Precision) using pycocotools.

Usage:
    uv run python evaluate.py --checkpoint output/checkpoint_best.pth
    uv run python evaluate.py --checkpoint output/checkpoint_epoch_10.pth --val_fold 5
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import tempfile

import torch
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import get_dataloader
from model import get_model, load_checkpoint, NUM_CLASSES, CLASS_NAMES
from config import DATABASE_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on Boxing Dataset")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--val_fold", type=int, default=5,
                        help="Validation fold number")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DALI worker threads")
    parser.add_argument("--score_threshold", type=float, default=0.05,
                        help="Score threshold for detections")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to JSON file")
    
    return parser.parse_args()


def get_device() -> torch.device:
    """Get the appropriate device for evaluation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, using CPU")
    return device


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_threshold: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Generate predictions for all images in the dataset.
    
    Returns:
        List of prediction dictionaries in COCO format
    """
    model.eval()
    predictions = []
    
    print("[INFO] Generating predictions...")
    
    for images, targets in tqdm(data_loader, desc="Inference"):
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            
            # Filter by score threshold
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            for box, score, label in zip(boxes, scores, labels):
                # Convert from [x1, y1, x2, y2] to [x, y, w, h] (COCO format)
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                predictions.append({
                    "image_id": int(image_id),
                    "category_id": int(label) - 1,  # Convert back to 0-indexed
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
    
    print(f"[INFO] Generated {len(predictions)} predictions")
    return predictions


def evaluate_coco(
    annotations_file: str,
    predictions: List[Dict[str, Any]],
    val_image_ids: List[int]
) -> Dict[str, float]:
    """
    Evaluate predictions using COCO metrics.
    
    Returns:
        Dictionary with mAP metrics
    """
    # Load ground truth
    coco_gt = COCO(annotations_file)
    
    # Filter to only validation images
    coco_gt.imgs = {k: v for k, v in coco_gt.imgs.items() if k in val_image_ids}
    coco_gt.anns = {k: v for k, v in coco_gt.anns.items() 
                    if v["image_id"] in val_image_ids}
    
    if len(predictions) == 0:
        print("[WARNING] No predictions to evaluate")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    
    # Save predictions to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name
    
    try:
        # Load predictions
        coco_dt = coco_gt.loadRes(pred_file)
        
        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            "mAP": coco_eval.stats[0],      # mAP @ IoU=0.5:0.95
            "mAP_50": coco_eval.stats[1],   # mAP @ IoU=0.5
            "mAP_75": coco_eval.stats[2],   # mAP @ IoU=0.75
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "AR_1": coco_eval.stats[6],     # AR @ max dets = 1
            "AR_10": coco_eval.stats[7],    # AR @ max dets = 10
            "AR_100": coco_eval.stats[8],   # AR @ max dets = 100
        }
    finally:
        os.unlink(pred_file)
    
    return metrics


def per_class_evaluation(
    annotations_file: str,
    predictions: List[Dict[str, Any]],
    val_image_ids: List[int]
) -> Dict[str, float]:
    """Compute per-class AP."""
    coco_gt = COCO(annotations_file)
    
    # Filter to validation images
    coco_gt.imgs = {k: v for k, v in coco_gt.imgs.items() if k in val_image_ids}
    coco_gt.anns = {k: v for k, v in coco_gt.anns.items()
                    if v["image_id"] in val_image_ids}
    
    if len(predictions) == 0:
        return {}
    
    # Save predictions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name
    
    try:
        coco_dt = coco_gt.loadRes(pred_file)
        
        per_class_ap = {}
        for cat_id in range(8):  # 8 boxing classes
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            
            # Get AP for this class
            ap = coco_eval.stats[0]  # mAP @ IoU=0.5:0.95
            class_name = CLASS_NAMES[cat_id + 1]  # +1 because index 0 is background
            per_class_ap[class_name] = ap
    finally:
        os.unlink(pred_file)
    
    return per_class_ap


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    
    annotations_file = f"{DATABASE_PATH}/annotations.json"
    
    print(f"\n{'='*60}")
    print("EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation fold: {args.val_fold}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"{'='*60}\n")
    
    # Create DALI data loader
    print("[INFO] Creating DALI data loader...")
    val_loader = get_dataloader(
        folds=[args.val_fold],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_training=False
    )
    
    # Get validation image IDs from the loader
    val_image_ids = [img["id"] for img in val_loader.images]
    print(f"[INFO] Evaluating on {len(val_image_ids)} images")
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model = model.to(device)
    print(f"[INFO] Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate predictions
    predictions = generate_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        score_threshold=args.score_threshold
    )
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = output_dir / "predictions.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"[INFO] Saved predictions to {pred_path}")
    
    # Run COCO evaluation
    print("\n" + "="*60)
    print("COCO EVALUATION RESULTS")
    print("="*60)
    
    metrics = evaluate_coco(annotations_file, predictions, val_image_ids)
    
    print(f"\nmAP @ IoU=0.5:0.95: {metrics['mAP']:.4f}")
    print(f"mAP @ IoU=0.5:      {metrics['mAP_50']:.4f}")
    print(f"mAP @ IoU=0.75:     {metrics['mAP_75']:.4f}")
    
    # Per-class evaluation
    print("\n" + "="*60)
    print("PER-CLASS AP @ IoU=0.5:0.95")
    print("="*60)
    
    per_class_ap = per_class_evaluation(annotations_file, predictions, val_image_ids)
    for class_name, ap in per_class_ap.items():
        print(f"{class_name}: {ap:.4f}")
    
    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "val_fold": args.val_fold,
        "num_images": len(val_image_ids),
        "num_predictions": len(predictions),
        "metrics": metrics,
        "per_class_ap": per_class_ap
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved results to {results_path}")
    
    print("\n[INFO] Evaluation complete!")


if __name__ == "__main__":
    main()
