import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)
import numpy as np
import glob
import cv2
import json
import shlex
import subprocess as sp
from config import DATABASE_PATH

def get_sorted_videos(path):
    videos = glob.glob(os.path.join(path, "*.MP4"))
    videos.sort()
    return videos


def video_stream_generator(video_files):
    """Yield frames sequentially from a list of video files."""
    for video in video_files:
        annotations = load_annotation(video)
        cam_name = video.split('/')[-2]
        video_name = video.split('/')[-1]
        frame_idx = 0

        print(f'read {video}')
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield video_name, cam_name, frame_idx, frame, annotations
            frame_idx += 1
        cap.release()
    print('[INFO] end of video stream')


def translate_label(label: str) -> str:
    LABEL_TRANSLATION = {
        "Głowa lewą ręką": "Head left hand",
        "Głowa prawą ręką": "Head right hand",
        "Korpus lewą ręką": "Body left hand",
        "Korpus prawą ręką": "Body right hand",
        "Blok lewą ręką": "Block left hand",
        "Blok prawą ręką": "Block right hand",
        "Chybienie lewą ręką": "Miss left hand",
        "Chybienie prawą ręką": "Miss right hand",
    }

    return LABEL_TRANSLATION.get(label, label)


def draw_info(video_name, cam_name, frame_idx, frame, annotations):
    """Draw bounding boxes and referee info on frame."""

    cv2.putText(frame, f'{cam_name}/{video_name} frame: {frame_idx}', (30, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if annotations is None:
        cv2.putText(frame, "Not reviewed by referee", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame

    # Draw tracks (bounding boxes)
    for track in annotations.get("tracks", []):
        for shape in track.get("shapes", []):
            if shape.get("frame") == frame_idx and shape.get("type") == 'rectangle' and not shape.get("outside"):
                x1, y1, x2, y2 = map(int, shape["points"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = translate_label(track.get("label", "Unknown"))
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw referee info
    reviewed_frames = annotations.get("frame_numbers_reviewed_by_referee", [])
    if frame_idx in reviewed_frames:
        cv2.putText(frame, "Reviewed by referee", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Not reviewed by referee", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


    return frame


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


def multi_camera_view(output_path="database_review.mp4", recording=False, debug=False):
    width, height = (1920, 1080)
    cam_dirs = [os.path.join(DATABASE_PATH, f"kam{i}") for i in range(1, 5)]
    video_lists = [get_sorted_videos(cam_dir) for cam_dir in cam_dirs]

    # Create frame generators for each camera
    streams = [video_stream_generator(videos) for videos in video_lists]
    target_size = (int(width / 2), int(height / 2))

    if recording:
        process = sp.Popen(shlex.split(f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {50} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 28 {output_path}'), stdin=sp.PIPE)

    while True:
        frames = []
        for gen in streams:
            try:
                video_name, cam_name, frame_idx, frame, annotations = next(gen)
                frame = draw_info(video_name, cam_name, frame_idx, frame, annotations)
            except StopIteration:
                print('[INFO] StopIteration, break loop')
                break
                # frame = np.zeros((width, height, 3), dtype=np.uint8)  # black placeholder if finished
            frames.append(frame)

        if len(frames) != 4:
            break

        frames = [cv2.resize(f, target_size) for f in frames]

        # Combine into 2x2 grid
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        combined = np.vstack((top, bottom))

        if recording:
            # Save to video
            process.stdin.write(combined.tobytes())

        if debug:
            # Show in window
            cv2.imshow("Multi-camera view", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # cv2.destroyAllWindows()

    print('[INFO] End of loop')
    if recording:
        print("[INFO] Closing ffmpeg process...")
        process.stdin.close()
        process.wait()
        process.terminate()
    print("[INFO] Done.")


if __name__ == "__main__":
    multi_camera_view(debug=True, recording=False)

