"""
Convert COCO JSON annotations to YOLO txt format, one directory per fold.

Output structure under yolo_dataset/:
    images/fold_1/  (symlinks to coco_images/*.jpg)
    labels/fold_1/  (generated .txt label files)
    images/fold_2/  ...
    ...

Run once before training:
    uv run --frozen python data-preparation/convert_coco_to_yolo.py
    uv run --frozen python data-preparation/convert_coco_to_yolo.py --folds 1 2 3 4 5
"""
import json
import os
import argparse
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '../../datasets/olympic-boxing-video-dataset'))


def convert_fold(fold_num: int, annotations_dir: str, images_dir: str, yolo_dir: str) -> int:
    ann_path = os.path.join(annotations_dir, f'annotations_fold_{fold_num}.json')
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f'Annotation file not found: {ann_path}')

    out_images = os.path.join(yolo_dir, 'images', f'fold_{fold_num}')
    out_labels = os.path.join(yolo_dir, 'labels', f'fold_{fold_num}')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_by_id = {img['id']: img for img in coco['images']}

    anns_by_image: dict = {}
    for ann in coco['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    for image_id, img_info in images_by_id.items():
        fname = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']

        src = os.path.abspath(os.path.join(images_dir, fname))
        dst = os.path.join(out_images, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)

        label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + '.txt')
        with open(label_path, 'w') as lf:
            for ann in anns_by_image.get(image_id, []):
                x_min, y_min, bbox_w, bbox_h = ann['bbox']
                class_id = ann['category_id']
                cx = max(0.0, min(1.0, (x_min + bbox_w / 2) / img_w))
                cy = max(0.0, min(1.0, (y_min + bbox_h / 2) / img_h))
                nw = max(0.0, min(1.0, bbox_w / img_w))
                nh = max(0.0, min(1.0, bbox_h / img_h))
                lf.write(f'{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')

    return len(images_by_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default=DATASET_DIR)
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    annotations_dir = os.path.join(args.dataset_dir, 'annotations')
    images_dir = os.path.join(args.dataset_dir, 'coco_images')
    yolo_dir = os.path.join(args.dataset_dir, 'yolo_dataset')

    print(f'Output: {yolo_dir}')
    for fold in args.folds:
        count = convert_fold(fold, annotations_dir, images_dir, yolo_dir)
        print(f'  fold_{fold}: {count} images')

    print('Done.')


if __name__ == '__main__':
    main()
