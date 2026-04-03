"""
Convert COCO JSON annotations to YOLO txt format and create the directory
structure expected by Ultralytics.

Creates:
    yolo_dataset/
    ├── images/
    │   ├── train/   (symlinks to original coco_images/*.jpg)
    │   └── val/     (symlinks to original coco_images/*.jpg)
    └── labels/
        ├── train/   (generated .txt label files)
        └── val/     (generated .txt label files)

YOLO label format (one line per object):
    class_id center_x center_y width height
    (all values normalized to [0, 1])

Usage:
    uv run convert_coco_to_yolo.py
    uv run convert_coco_to_yolo.py --dataset-dir ../datasets/olympic-boxing-video-dataset
"""
import json
import os
import argparse
from pathlib import Path


def convert_coco_to_yolo_labels(coco_json_path: str, images_dir: str,
                                 output_images_dir: str, output_labels_dir: str) -> int:
    """
    Convert a single COCO annotation file to YOLO format labels.

    Args:
        coco_json_path: Path to the COCO JSON annotation file
        images_dir: Path to the directory containing the original images
        output_images_dir: Path to create symlinks for images (e.g., yolo_dataset/images/train)
        output_labels_dir: Path to write YOLO label files (e.g., yolo_dataset/labels/train)

    Returns:
        Number of images processed
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Build lookup: image_id -> image info
    images_by_id = {img['id']: img for img in coco_data['images']}

    # Build lookup: image_id -> list of annotations
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    processed = 0
    for image_id, image_info in images_by_id.items():
        file_name = image_info['file_name']
        img_w = image_info['width']
        img_h = image_info['height']

        # Create symlink for the image
        src_image_path = os.path.abspath(os.path.join(images_dir, file_name))
        dst_image_path = os.path.join(output_images_dir, file_name)

        if not os.path.exists(dst_image_path):
            os.symlink(src_image_path, dst_image_path)

        # Create YOLO label file
        label_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_name)

        annotations = annotations_by_image.get(image_id, [])

        with open(label_path, 'w') as lf:
            for ann in annotations:
                # COCO bbox: [x_min, y_min, width, height] (absolute pixels)
                x_min, y_min, bbox_w, bbox_h = ann['bbox']
                class_id = ann['category_id']

                # Convert to YOLO format: center_x, center_y, width, height (normalized)
                center_x = (x_min + bbox_w / 2) / img_w
                center_y = (y_min + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h

                # Clamp to [0, 1]
                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                lf.write(f'{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n')

        processed += 1

    return processed


def prepare_yolo_dataset(dataset_dir: str, train_folds: list[int], val_fold: int):
    """
    Convert COCO fold annotations to YOLO format and create the full directory structure.

    Args:
        dataset_dir: Root dataset directory (e.g., ../datasets/olympic-boxing-video-dataset)
        train_folds: List of fold numbers for training (e.g., [1, 2, 3, 4])
        val_fold: Fold number for validation (e.g., 5)
    """
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    images_dir = os.path.join(dataset_dir, 'coco_images')
    yolo_dir = os.path.join(dataset_dir, 'yolo_dataset')

    print(f'Preparing YOLO dataset in: {yolo_dir}')
    print(f'Source images: {images_dir}')
    print(f'Source annotations: {annotations_dir}')

    # First, merge training fold annotations in memory and convert
    print(f'\n--- Training set (folds {train_folds}) ---')
    train_images_dir = os.path.join(yolo_dir, 'images', 'train')
    train_labels_dir = os.path.join(yolo_dir, 'labels', 'train')

    total_train = 0
    for fold_num in train_folds:
        fold_path = os.path.join(annotations_dir, f'annotations_fold_{fold_num}.json')
        if not os.path.exists(fold_path):
            raise FileNotFoundError(f'Annotation file not found: {fold_path}')

        count = convert_coco_to_yolo_labels(fold_path, images_dir, train_images_dir, train_labels_dir)
        print(f'  Fold {fold_num}: {count} images')
        total_train += count

    print(f'  Total training: {total_train} images')

    # Convert validation fold
    print(f'\n--- Validation set (fold {val_fold}) ---')
    val_images_dir = os.path.join(yolo_dir, 'images', 'val')
    val_labels_dir = os.path.join(yolo_dir, 'labels', 'val')

    val_path = os.path.join(annotations_dir, f'annotations_fold_{val_fold}.json')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f'Annotation file not found: {val_path}')

    val_count = convert_coco_to_yolo_labels(val_path, images_dir, val_images_dir, val_labels_dir)
    print(f'  Validation: {val_count} images')

    print(f'\nYOLO dataset ready at: {yolo_dir}')
    print(f'  images/train: {total_train} images (symlinks)')
    print(f'  images/val:   {val_count} images (symlinks)')
    print(f'  labels/train: {total_train} label files')
    print(f'  labels/val:   {val_count} label files')


def main():
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    parser.add_argument(
        '--dataset-dir',
        default='../datasets/olympic-boxing-video-dataset',
        help='Root dataset directory (default: ../datasets/olympic-boxing-video-dataset)'
    )
    parser.add_argument(
        '--train-folds',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='Fold numbers for training (default: 1 2 3 4)'
    )
    parser.add_argument(
        '--val-fold',
        type=int,
        default=5,
        help='Fold number for validation (default: 5)'
    )

    args = parser.parse_args()
    prepare_yolo_dataset(args.dataset_dir, args.train_folds, args.val_fold)


if __name__ == '__main__':
    main()
