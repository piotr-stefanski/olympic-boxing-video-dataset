import cv2
import glob
import os
import json
import argparse
from tqdm import tqdm
from config import EXPECTED_TOTAL_NUMBER_OF_FRAMES, FOLDS_DEFINITION, DATABASE_PATH, COCO_IMAGES_DIR_PATH, COCO_ANNOTATIONS_DIR_PATH

LABEL_TO_INDEX_MAP = {
    'Głowa lewą ręką': 0,
    'Głowa prawą ręką': 1,
    'Korpus lewą ręką': 2,
    'Korpus prawą ręką': 3,
    'Blok lewą ręką': 4,
    'Blok prawą ręką': 5,
    'Chybienie lewą ręką': 6,
    'Chybienie prawą ręką': 7,

}

def get_info():
    return {
        "year": "2025",
        "version": "1",
        "description": "Boxing fight database",
        "contributor": "Piotr Stefański",
        "url": "",
        "date_created": "2025-12-07T09:30:00+00:00"
    }

def get_licenses():
    return [{
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/deed",
            "name": "CC BY 4.0"
        }]

def get_categories():
    return [
        {
            "id": 0,
            "name": "Punch to the head with the left hand",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "Punch to the head with the right hand",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "Punch to the torso with the left hand",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "Punch to the torso with the right hand",
            "supercategory": "none"
        },
        {
            "id": 4,
            "name": "Block with the left hand",
            "supercategory": "none"
        },
        {
            "id": 5,
            "name": "Block with the right hand",
            "supercategory": "none"
        },
        {
            "id": 6,
            "name": "Missed punch with the left hand",
            "supercategory": "none"
        },
        {
            "id": 7,
            "name": "Missed punch with the right hand",
            "supercategory": "none"
        },
    ]

def load_annotation(video_file):
    base, _ = os.path.splitext(video_file)
    ann_path = f"{base}_annotations.json"
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r") as f:
                annotations = json.load(f)
            return annotations
        except Exception as e:
            print(f"[ERROR] Failed to read annotation for {video_file}: {e}")
            return None
    else:
        print(f"[INFO] Video file {os.path.basename(video_file)} not annotated")
        return None

def get_sorted_videos(path):
    videos = glob.glob(f'{path}/*.MP4')
    videos.sort()
    return videos

def get_label_by_category_id(category_id):
    categories = get_categories()

    # 2. Iterate to find the matching ID
    for cat in categories:
        if cat["id"] == category_id:
            return cat["name"]

    return "Unknown Category"


def draw_annotation_on_frame(frame, coco_annotation):
    x, y, w, h = coco_annotation['bbox']
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    label = get_label_by_category_id(coco_annotation['category_id'])
    draw_frame = frame.copy()

    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(draw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return draw_frame

def is_frame_was_reviewed_by_referee(video_frame_idx, annotations):
    if annotations is None:
        return False
    return video_frame_idx in annotations.get("frame_numbers_reviewed_by_referee")

def get_annotated_frame_indices(annotations):
    if annotations is None:
        return set()
    return {
        shape["frame"]
        for track in annotations.get("tracks", [])
        for shape in track.get("shapes", [])
        if shape.get("type") == 'rectangle' and not shape.get("outside")
    }

def collect_and_save_annotations(fold_data):
    for fold_name, fold in fold_data.items():
        data = {
            "info": get_info(),
            "licenses": get_licenses(),
            "categories": get_categories(),
            "images": fold["images"],
            "annotations": fold["annotations"]
        }

        output_path = f'{COCO_ANNOTATIONS_DIR_PATH}/annotations_{fold_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'[INFO] Saved {len(fold["images"])} images and {len(fold["annotations"])} annotations to {output_path}')

def get_frame_annotations(annotations_idx, global_image_idx, video_frame_idx, annotations):
    if annotations is None:
        return annotations_idx, []

    frame_annotations = []

    # Draw tracks (bounding boxes)
    for track in annotations.get("tracks", []):
        for shape in track.get("shapes", []):
            if shape.get("frame") == video_frame_idx and shape.get("type") == 'rectangle' and not shape.get("outside"):
                x_min, y_min, x_max, y_max = map(int, shape["points"])
                width = x_max-x_min
                height = y_max-y_min
                area = width * height
                bbox = [x_min, y_min, width, height]
                label_id = LABEL_TO_INDEX_MAP[track.get("label")]

                frame_annotations.append({
                    "id": annotations_idx,
                    "image_id": global_image_idx,
                    "category_id": label_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0
                })

    return frame_annotations

def get_coco_image_based_on_frame(global_image_idx, cam_frame_idx, cam_name, fold_number, frame):
    h, w, _ = frame.shape
    return {
            "id": global_image_idx,
            "license": 1,
            "file_name": f"{global_image_idx}.jpg",
            "height": h,
            "width": w,
            "date_captured": "2021-02-27T10:00:00+00:00",
            "cam_frame_id": cam_frame_idx,
            "cam_name": cam_name,
            "fold_number": fold_number
        }

def save_coco_image(image, image_id, save_images, target_shape: dict|None = None):
    if save_images:
        # TODO: IMPORTANT: implement also scaling annotations when shape of image is changed
        if target_shape is not None:
            image = cv2.resize(image, target_shape)

        cv2.imwrite(f'{COCO_IMAGES_DIR_PATH}/{image_id}.jpg', image)

def get_fold_number_based_on_frame_idx(cam_frame_idx: int) -> str:
    for fold_name, ranges in FOLDS_DEFINITION.items():
        # Check if the index is within the start and end (inclusive)
        if ranges['start'] <= cam_frame_idx <= ranges['end']:
            return fold_name

    raise ValueError("Index out of bounds")

def main(debug=False, save_images=False, target_shape=None):
    global_image_idx = 0
    annotations_idx = 0
    fold_data = {fold: {"images": [], "annotations": []} for fold in FOLDS_DEFINITION}

    for cam_number in [2, 4]:
        cam_name = f'kam{cam_number}'
        video_paths = get_sorted_videos(f'{DATABASE_PATH}/{cam_name}')
        cam_frame_idx = 0

        for video_path in video_paths:
            annotations = load_annotation(video_path)
            annotated_frames = get_annotated_frame_indices(annotations)

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for video_frame_idx in tqdm(range(total_frames), desc=f'{cam_name} - {os.path.basename(video_path)}'):
                ret, frame = cap.read()
                if not ret:
                    break

                if cam_frame_idx >= FOLDS_DEFINITION['fold_1']['start'] and video_frame_idx in annotated_frames: # you can also only check whether a frame has been reviewed (the frame has been viewed, but no bbox has been drawn) by is_frame_was_reviewed_by_referee function, but then you get > 300k examples and need 156GB of disk space
                    fold_number = get_fold_number_based_on_frame_idx(cam_frame_idx)
                    image = get_coco_image_based_on_frame(global_image_idx, cam_frame_idx, cam_name, fold_number, frame)
                    fold_data[fold_number]["images"].append(image)
                    save_coco_image(frame, global_image_idx, save_images, target_shape=target_shape)

                    frame_annotations = get_frame_annotations(annotations_idx, global_image_idx, video_frame_idx, annotations)
                    if frame_annotations is not None:
                        fold_data[fold_number]["annotations"].extend(frame_annotations)

                        if annotations_idx%1000 == 0 and len(frame_annotations) > 0 and debug:
                            draw_frame = draw_annotation_on_frame(frame, frame_annotations[0])
                            cv2.imshow('frame', draw_frame)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        annotations_idx += 1

                    global_image_idx += 1

                cam_frame_idx += 1
            cap.release()

            collect_and_save_annotations(fold_data)


        print(f'[INFO] end of videos for {cam_name}')
        if cam_frame_idx != EXPECTED_TOTAL_NUMBER_OF_FRAMES:
            print(f'[ERROR !!!]: wrong number of frames for {cam_name}, {cam_frame_idx=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-images', help='Description for foo argument', action='store_true')
    parser.add_argument('--target-shape', help='Target shape (width height) for resizing saved images', type=int, nargs=2, default=None, metavar=('W', 'H'))
    args = parser.parse_args()

    target_shape = tuple(args.target_shape) if args.target_shape else None

    os.makedirs(COCO_ANNOTATIONS_DIR_PATH, exist_ok=True)

    if args.save_images:
        os.makedirs(COCO_IMAGES_DIR_PATH, exist_ok=True)

    main(save_images=args.save_images, target_shape=target_shape)
