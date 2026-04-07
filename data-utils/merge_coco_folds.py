"""
Merge multiple COCO annotation fold files into a single annotation file.

This is needed for Ultralytics training, which expects one annotation file per split.
Usage:
    uv run merge_coco_folds.py
    uv run merge_coco_folds.py --folds 1 2 3 4 --output annotations_train.json
"""
import json
import argparse
import os


def merge_coco_folds(annotation_dir: str, fold_numbers: list[int], output_filename: str) -> str:
    """
    Merge multiple COCO annotation files into a single file.

    Args:
        annotation_dir: Directory containing the fold annotation files
        fold_numbers: List of fold numbers to merge (e.g., [1, 2, 3, 4])
        output_filename: Name of the output merged annotation file

    Returns:
        Path to the merged annotation file
    """
    merged_images = []
    merged_annotations = []
    shared_info = None
    shared_licenses = None
    shared_categories = None

    for fold_num in fold_numbers:
        fold_path = os.path.join(annotation_dir, f'annotations_fold_{fold_num}.json')
        if not os.path.exists(fold_path):
            raise FileNotFoundError(f'Annotation file not found: {fold_path}')

        with open(fold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Preserve shared metadata from the first fold
        if shared_info is None:
            shared_info = data['info']
            shared_licenses = data['licenses']
            shared_categories = data['categories']

        merged_images.extend(data['images'])
        merged_annotations.extend(data['annotations'])

        print(f'  Fold {fold_num}: {len(data["images"])} images, {len(data["annotations"])} annotations')

    merged_data = {
        'info': shared_info,
        'licenses': shared_licenses,
        'categories': shared_categories,
        'images': merged_images,
        'annotations': merged_annotations,
    }

    output_path = os.path.join(annotation_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f'Merged {len(merged_images)} images and {len(merged_annotations)} annotations')
    print(f'Saved to: {output_path}')

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Merge COCO annotation folds into a single file')
    parser.add_argument(
        '--annotations-dir',
        default='../datasets/olympic-boxing-video-dataset/annotations',
        help='Directory containing fold annotation files'
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='Fold numbers to merge (default: 1 2 3 4)'
    )
    parser.add_argument(
        '--output',
        default='annotations_train.json',
        help='Output filename (default: annotations_train.json)'
    )

    args = parser.parse_args()

    print(f'Merging folds {args.folds} from {args.annotations_dir}')
    merge_coco_folds(args.annotations_dir, args.folds, args.output)


if __name__ == '__main__':
    main()
