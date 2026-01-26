"""
Faster R-CNN model for boxing action detection.

This module provides a pre-configured Faster R-CNN model with ResNet-50-FPN backbone
for detecting 8 classes of boxing actions.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Optional, Tuple

# Number of boxing action classes (8 actions + 1 background)
NUM_CLASSES = 9  # 8 boxing actions + background

# Class names for visualization/logging
CLASS_NAMES = [
    "background",
    "Punch to the head with the left hand",
    "Punch to the head with the right hand", 
    "Punch to the torso with the left hand",
    "Punch to the torso with the right hand",
    "Block with the left hand",
    "Block with the right hand",
    "Missed punch with the left hand",
    "Missed punch with the right hand",
]


def get_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    trainable_backbone_layers: int = 3,
) -> torch.nn.Module:
    """
    Get a Faster R-CNN model configured for boxing action detection.
    
    Args:
        num_classes: Number of classes (including background). Default: 9
        pretrained: Whether to use COCO pretrained weights. Default: True
        pretrained_backbone: Whether backbone uses ImageNet pretrained weights. Default: True
        min_size: Minimum size of the image to be rescaled before feeding to backbone.
        max_size: Maximum size of the image to be rescaled before feeding to backbone.
        trainable_backbone_layers: Number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all layers are trainable.
    
    Returns:
        Configured Faster R-CNN model
    """
    if pretrained:
        # Load model with COCO pretrained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size
        )
    else:
        # Load model without pretrained weights
        backbone_weights = "IMAGENET1K_V1" if pretrained_backbone else None
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size
        )
    
    # Replace the classifier head with one that has correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def print_model_summary(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int] = (3, 800, 1333),
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    Print a detailed summary of the model architecture.
    
    Args:
        model: The model to summarize
        input_size: Input tensor size (C, H, W)
        batch_size: Batch size for the summary
        device: Device to run the model on
    """
    try:
        from torchinfo import summary
        
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_size).to(device)
        
        print("=" * 80)
        print("MODEL SUMMARY: Faster R-CNN with ResNet-50-FPN Backbone")
        print("=" * 80)
        print(f"Number of classes: {NUM_CLASSES}")
        print(f"Class names: {CLASS_NAMES}")
        print("=" * 80)
        
        # Note: Faster R-CNN expects list of images, not batched tensor
        # We'll just count parameters instead for the summary
        summary(
            model,
            input_data=[[dummy_input[0]]],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
            verbose=1
        )
        
    except ImportError:
        print("[WARNING] torchinfo not installed. Install with: pip install torchinfo")
        print("\nModel Architecture (basic info):")
        print("-" * 40)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze all backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("[INFO] Backbone frozen - only head is trainable")


def unfreeze_backbone(model: torch.nn.Module) -> None:
    """Unfreeze all backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("[INFO] Backbone unfrozen - all layers are trainable")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.nn.Module, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load weights into. If None, creates new model.
        device: Device to load model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is None:
        model = get_model()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")
    
    return model, checkpoint


if __name__ == "__main__":
    print("Testing Faster R-CNN Model for Boxing Detection")
    print("-" * 50)
    
    # Create model
    model = get_model()
    print(f"Model created successfully!")
    
    # Print summary
    print_model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Create dummy input (list of images as expected by Faster R-CNN)
    dummy_images = [torch.randn(3, 800, 1200).to(device)]
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"Output keys: {outputs[0].keys()}")
    print(f"Number of detections: {len(outputs[0]['boxes'])}")
    print(f"Boxes shape: {outputs[0]['boxes'].shape}")
    print(f"Labels shape: {outputs[0]['labels'].shape}")
    print(f"Scores shape: {outputs[0]['scores'].shape}")
    
    print("\n✓ Model test passed!")
