"""
Training script for Faster R-CNN on Olympic Boxing Dataset.

This script handles the complete training pipeline including:
- Data loading with fold-based splitting
- Model initialization and checkpoint loading
- Training loop with loss tracking
- Learning rate scheduling
- Checkpoint saving
- TensorBoard logging

Usage:
    # Basic training
    uv run python train.py --epochs 10 --batch_size 4

    # Resume from checkpoint
    uv run python train.py --resume output/checkpoint_epoch_5.pth

    # Custom fold configuration
    uv run python train.py --train_folds 1,2,3,4 --val_fold 5
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from model import get_model, print_model_summary, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Boxing Dataset")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    
    # Learning rate scheduler
    parser.add_argument("--lr_step_size", type=int, default=3, help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    
    # Data configuration
    parser.add_argument("--train_folds", type=str, default="1,2,3,4", 
                        help="Comma-separated list of training fold numbers")
    parser.add_argument("--val_fold", type=int, default=5, help="Validation fold number")
    parser.add_argument("--num_workers", type=int, default=4, help="DALI worker threads")
    
    # Model configuration
    parser.add_argument("--min_size", type=int, default=800, help="Min image size")
    parser.add_argument("--max_size", type=int, default=1333, help="Max image size")
    parser.add_argument("--trainable_backbone_layers", type=int, default=3,
                        help="Number of trainable backbone layers (0-5)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (iterations)")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Debug
    parser.add_argument("--print_model", action="store_true", help="Print model summary")
    parser.add_argument("--dry_run", action="store_true", help="Run one iteration only (for testing)")
    
    return parser.parse_args()


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory and return Path object."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create runs directory for TensorBoard
    runs_path = output_path / "runs"
    runs_path.mkdir(exist_ok=True)
    
    return output_path


def get_device() -> torch.device:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, using CPU")
    return device


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_freq: int = 10,
    dry_run: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with average losses for the epoch
    """
    model.train()
    
    # Loss accumulators
    loss_accum = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total_loss": 0.0
    }
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move data to device (non_blocking for async transfer)
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate losses
        loss_accum["total_loss"] += losses.item()
        for key in loss_dict:
            if key in loss_accum:
                loss_accum[key] += loss_dict[key].item()
        num_batches += 1
        
        # Calculate global step for TensorBoard
        global_step = epoch * len(data_loader) + batch_idx
        
        # Log to TensorBoard
        if batch_idx % log_freq == 0:
            writer.add_scalar("Loss/train_total", losses.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f"Loss/train_{key}", value.item(), global_step)
            
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                  f"Time: {elapsed:.1f}s")
        
        if dry_run:
            print("[DRY RUN] Stopping after one iteration")
            break
    
    # Calculate averages
    for key in loss_accum:
        loss_accum[key] /= max(num_batches, 1)
    
    return loss_accum


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """
    Validate the model on the validation set.
    
    Note: Faster R-CNN in eval mode doesn't return losses, only predictions.
    For proper mAP evaluation, use evaluate.py with pycocotools.
    Here we just run inference and count detections.
    
    Returns:
        Average number of detections per image
    """
    model.eval()
    
    total_detections = 0
    num_images = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        
        outputs = model(images)
        
        for output in outputs:
            # Count detections with score > 0.5
            high_conf = output["scores"] > 0.5
            total_detections += high_conf.sum().item()
            num_images += 1
        
        if batch_idx % 50 == 0:
            print(f"Validation batch [{batch_idx}/{len(data_loader)}]")
    
    avg_detections = total_detections / max(num_images, 1)
    writer.add_scalar("Val/avg_detections", avg_detections, epoch)
    
    print(f"Validation: {num_images} images, {total_detections} high-conf detections, "
          f"avg {avg_detections:.2f} per image")
    
    return avg_detections


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    loss: float,
    output_dir: Path,
    is_best: bool = False
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    # Save latest checkpoint (for easy resume)
    latest_path = output_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = output_dir / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)
        print(f"[INFO] Saved best checkpoint: {best_path}")


def main():
    args = parse_args()
    
    # Setup
    output_dir = setup_output_dir(args.output_dir)
    device = get_device()
    
    # Parse fold configuration
    train_folds = [int(f) for f in args.train_folds.split(",")]
    val_folds = [args.val_fold]
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Train folds: {train_folds}")
    print(f"Validation fold: {val_folds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create data loaders using NVIDIA DALI
    print("[INFO] Creating DALI data loaders...")
    
    train_loader = get_dataloader(
        folds=train_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_training=True
    )
    val_loader = get_dataloader(
        folds=val_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_training=False
    )
    
    # Create model
    print("[INFO] Creating model...")
    model = get_model(
        num_classes=NUM_CLASSES,
        min_size=args.min_size,
        max_size=args.max_size,
        trainable_backbone_layers=args.trainable_backbone_layers
    )
    model = model.to(device)
    
    if args.print_model:
        print_model_summary(model)
    
    # Create optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")
    
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("loss", float("inf"))
        print(f"[INFO] Resumed from epoch {start_epoch}")
    
    # Create TensorBoard writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=output_dir / "runs" / timestamp)
    
    # Log hyperparameters
    writer.add_hparams(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "train_folds": args.train_folds,
            "val_fold": args.val_fold,
        },
        {}
    )
    
    # Training loop
    print(f"\n[INFO] Starting training from epoch {start_epoch} to {args.epochs}")
    print(f"[INFO] TensorBoard logs at: {output_dir / 'runs' / timestamp}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            writer=writer,
            log_freq=args.log_freq,
            dry_run=args.dry_run
        )
        
        # Log epoch-level metrics
        writer.add_scalar("Loss/epoch_train_total", train_losses["total_loss"], epoch)
        
        # Step scheduler
        scheduler.step()
        writer.add_scalar("LR/learning_rate", scheduler.get_last_lr()[0], epoch)
        
        # Validate
        if not args.dry_run:
            validate(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                writer=writer
            )
        
        # Save checkpoint
        is_best = train_losses["total_loss"] < best_loss
        if is_best:
            best_loss = train_losses["total_loss"]
        
        if (epoch + 1) % args.checkpoint_freq == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=train_losses["total_loss"],
                output_dir=output_dir,
                is_best=is_best
            )
        
        if args.dry_run:
            print("[DRY RUN] Stopping after one epoch")
            break
    
    writer.close()
    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Best loss: {best_loss:.4f}")
    print(f"[INFO] Checkpoints saved to: {output_dir}")
    print(f"[INFO] TensorBoard logs: tensorboard --logdir {output_dir / 'runs'}")


if __name__ == "__main__":
    main()
