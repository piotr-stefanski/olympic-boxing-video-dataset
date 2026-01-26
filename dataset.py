"""
High-performance dataset for Olympic Boxing Video Dataset.

Uses NVIDIA DALI for GPU-accelerated image decoding and augmentation.
Supports fold-based data splitting for cross-validation.
"""

import os
import json
import torch
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    from nvidia.dali import pipeline_def, fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    raise ImportError(
        "NVIDIA DALI is required for this dataset. "
        "Please install nvidia-dali-cuda120: pip install nvidia-dali-cuda120"
    )

from config import DATABASE_PATH, FOLDS_DEFINITION, COCO_IMAGES_DIR_PATH


# ============================================================================
# NVIDIA DALI Pipeline for high-performance data loading
# ============================================================================

@pipeline_def
def boxing_dali_pipeline(
    file_list: str,
    images_dir: str,
    shard_id: int = 0,
    num_shards: int = 1,
    random_shuffle: bool = True,
    is_training: bool = True
):
    """
    DALI pipeline for GPU-accelerated image loading and augmentation.
    
    Note: batch_size, num_threads, and device_id are handled by @pipeline_def decorator.
    
    Args:
        file_list: Path to file containing image paths (one per line)
        images_dir: Base directory for images
        shard_id: Shard ID for distributed training
        num_shards: Total number of shards
        random_shuffle: Whether to shuffle data
        is_training: Whether this is a training pipeline
    """
    # Read file paths from list
    jpegs, labels = fn.readers.file(
        file_root=images_dir,
        file_list=file_list,
        random_shuffle=random_shuffle,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader"
    )
    
    # Decode images on GPU
    images = fn.decoders.image(
        jpegs,
        device="mixed",  # Decode on GPU
        output_type=types.RGB
    )
    
    if is_training:
        # Random horizontal flip
        images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
        
        # Color augmentation
        images = fn.color_twist(
             images,
             brightness=fn.random.uniform(range=(0.8, 1.2)),
             contrast=fn.random.uniform(range=(0.8, 1.2)),
             saturation=fn.random.uniform(range=(0.8, 1.2))
        )
    
    # Normalize to [0, 1] range
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0]
    )
    
    return images, labels


class DALIBoxingLoader:
    """
    DALI-based data loader for boxing dataset.
    
    Provides PyTorch-compatible iterator with GPU-accelerated loading.
    Note: This loader returns only images. Annotations must be handled separately.
    """
    
    def __init__(
        self,
        folds: List[int] = [1, 2, 3, 4],
        batch_size: int = 4,
        num_threads: int = 4,
        device_id: int = 0,
        is_training: bool = True,
        annotations_file: Optional[str] = None,
        images_dir: Optional[str] = None
    ):
        self.batch_size = batch_size
        self.annotations_file = annotations_file or f"{DATABASE_PATH}/annotations.json"
        self.images_dir = images_dir or COCO_IMAGES_DIR_PATH
        self.is_training = is_training
        
        # Load annotations to filter by fold
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        self.num_classes = len(self.categories)
        
        fold_names = [f"fold_{i}" for i in folds]
        self.images = [
            img for img in coco_data['images']
            if img.get('fold_number') in fold_names
        ]
        
        # Build annotation lookup
        self.img_to_anns: Dict[int, List[Dict]] = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Create file list for DALI
        self.file_list_path = f"/tmp/dali_filelist_{device_id}_{folds[0]}.txt"
        with open(self.file_list_path, 'w') as f:
            for img in self.images:
                f.write(f"{img['file_name']} {img['id']}\n")
        
        # Build DALI pipeline
        self.pipeline = boxing_dali_pipeline(
            file_list=self.file_list_path,
            images_dir=self.images_dir,
            device_id=device_id,
            batch_size=batch_size,
            num_threads=num_threads,
            random_shuffle=is_training,
            is_training=is_training
        )
        self.pipeline.build()
        
        self.size = len(self.images)
        print(f"[INFO] DALIBoxingLoader initialized with {self.size} images from folds {folds}")
        print(f"[INFO] Number of classes: {self.num_classes}")
    
    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate through batches, yielding (images, targets) pairs."""
        self.pipeline.reset()
        
        for _ in range(len(self)):
            output = self.pipeline.run()
            images = output[0].as_cpu() if hasattr(output[0], 'as_cpu') else output[0]
            image_ids = output[1].as_cpu() if hasattr(output[1], 'as_cpu') else output[1]
            
            # Convert to PyTorch tensors
            batch_images = []
            batch_targets = []
            
            for i in range(len(images)):
                img_tensor = torch.from_numpy(np.array(images.at(i)))
                img_id = int(image_ids.at(i))
                
                # Get annotations for this image
                anns = self.img_to_anns.get(img_id, [])
                
                boxes = []
                labels = []
                areas = []
                iscrowd = []
                
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'] + 1)
                    areas.append(ann['area'])
                    iscrowd.append(ann.get('iscrowd', 0))
                
                if len(boxes) > 0:
                    target = {
                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                        'labels': torch.tensor(labels, dtype=torch.int64),
                        'image_id': torch.tensor([img_id], dtype=torch.int64),
                        'area': torch.tensor(areas, dtype=torch.float32),
                        'iscrowd': torch.tensor(iscrowd, dtype=torch.int64)
                    }
                else:
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'labels': torch.zeros((0,), dtype=torch.int64),
                        'image_id': torch.tensor([img_id], dtype=torch.int64),
                        'area': torch.zeros((0,), dtype=torch.float32),
                        'iscrowd': torch.zeros((0,), dtype=torch.int64)
                    }
                
                batch_images.append(img_tensor)
                batch_targets.append(target)
            
            yield batch_images, batch_targets
    
    def get_category_names(self) -> Dict[int, str]:
        """Return mapping of category IDs to names."""
        return {cat_id + 1: cat['name'] for cat_id, cat in self.categories.items()}


def get_dataloader(
    folds: List[int],
    batch_size: int = 4,
    num_workers: int = 4,
    device_id: int = 0,
    is_training: bool = True,
    **kwargs
):
    """
    Factory function to get DALI data loader.
    
    Args:
        folds: List of fold numbers to include
        batch_size: Batch size
        num_workers: Number of worker threads
        device_id: GPU device ID for DALI
        is_training: Whether this is for training
        **kwargs: Additional arguments passed to DALIBoxingLoader
    
    Returns:
        DALIBoxingLoader instance
    """
    return DALIBoxingLoader(
        folds=folds,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=device_id,
        is_training=is_training,
        **kwargs
    )


if __name__ == "__main__":
    # Test the DALI loader
    print("Testing DALIBoxingLoader...")
    loader = DALIBoxingLoader(folds=[1], batch_size=2, num_threads=2)
    
    for images, targets in loader:
        print(f"Batch size: {len(images)}")
        print(f"First image shape: {images[0].shape}")
        print(f"First target boxes: {targets[0]['boxes'].shape}")
        print(f"Categories: {loader.get_category_names()}")
        break
