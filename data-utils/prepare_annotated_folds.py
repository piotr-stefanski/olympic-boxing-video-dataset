"""
Filter annotations.json to annotated-only images and split into per-fold files.

Output: datasets/olympic-boxing-video-dataset/annotations/annotations_fold_N.json
        datasets/olympic-boxing-video-dataset/coco_images -> symlink to database/coco_images

Usage:
    python prepare_annotated_folds.py
    python prepare_annotated_folds.py --input annotations.json --output-dir ../datasets/olympic-boxing-video-dataset/annotations
"""
import json
import os
import argparse
from collections import defaultdict

INPUT_FILE = "annotations.json"
OUTPUT_DIR = "../datasets/olympic-boxing-video-dataset/annotations"
COCO_IMAGES_SRC = os.path.abspath("coco_images")
COCO_IMAGES_DST = "../datasets/olympic-boxing-video-dataset/coco_images"


def main(input_file: str, output_dir: str):
    print(f"Loading {input_file}...")
    with open(input_file, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    print(f"Total images (reviewed): {len(images)}")
    print(f"Total annotations: {len(annotations)}")

    # Find image IDs that have at least one annotation
    annotated_image_ids = {ann["image_id"] for ann in annotations}
    print(f"Images with annotations: {len(annotated_image_ids)}")

    # Filter images to annotated-only, group by fold
    folds: dict[str, list] = defaultdict(list)
    for img in images:
        if img["id"] in annotated_image_ids:
            fold = img.get("fold_number")
            if fold is None:
                print(f"[WARN] Image {img['id']} has no fold_number, skipping")
                continue
            folds[fold].append(img)

    # Group annotations by image_id for quick lookup
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Write per-fold files
    os.makedirs(output_dir, exist_ok=True)

    total_written_images = 0
    total_written_annotations = 0

    for fold_name in sorted(folds.keys()):
        fold_images = folds[fold_name]
        fold_annotations = []
        for img in fold_images:
            fold_annotations.extend(anns_by_image[img["id"]])

        # Reassign annotation ids to guarantee uniqueness within this fold file
        for new_id, ann in enumerate(fold_annotations, start=1):
            ann["id"] = new_id

        fold_num = fold_name.split("_")[-1]  # "fold_1" -> "1"
        output_path = os.path.join(output_dir, f"annotations_fold_{fold_num}.json")

        data = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "categories": coco["categories"],
            "images": fold_images,
            "annotations": fold_annotations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        print(f"  {fold_name}: {len(fold_images):>5} images, {len(fold_annotations):>5} annotations -> {output_path}")
        total_written_images += len(fold_images)
        total_written_annotations += len(fold_annotations)

    print(f"\nTotal written: {total_written_images} images, {total_written_annotations} annotations")

    # Set up coco_images symlink
    dst = COCO_IMAGES_DST
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(COCO_IMAGES_SRC, dst)
        print(f"Created symlink: {dst} -> {COCO_IMAGES_SRC}")
    else:
        print(f"Symlink already exists: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.input, args.output_dir)
