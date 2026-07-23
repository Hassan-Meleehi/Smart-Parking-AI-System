#!/usr/bin/env python
import os
import time
import sys
import subprocess

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ------------- CONFIG -------------

IMAGE_URL = os.environ.get(
    "IMAGE_URL",
    ""
)
FIREBASE_DB_URL = os.environ.get(
    "FIREBASE_DB_URL",
    ""
).rstrip("/")

YOLO_MODEL = os.environ.get(
    "YOLO_MODEL",
    "yolov8n-seg.pt"
)

# Where to save raw collision snapshots
COLLISION_IMG_DIR = os.environ.get(
    "COLLISION_IMG_DIR",
    "/home/hassanm/smart-parking-edge/collisions"
)

os.makedirs(COLLISION_IMG_DIR, exist_ok=True)

# IoU threshold to consider “collision”
COLLISION_IOU_THRESH = 0.35
# seconds between collision events (cooldown)
COLLISION_COOLDOWN = 8.0

# ------------- HELPERS -------------

def fetch_image(url: str, timeout: int = 5):
    for _ in range(3):
        try:
            r = requests.get(url, params={"t": time.time()}, timeout=timeout, stream=True)
            r.raise_for_status()
            data = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
        except Exception as e:
            print("[collision] fetch error:", e)
        time.sleep(1)
    return None


def iou(boxA, boxB):
    (x1, y1, x2, y2) = boxA
    (x1b, y1b, x2b, y2b) = boxB

    xA = max(x1, x1b)
    yA = max(y1, y1b)
    xB = min(x2, x2b)
    yB = min(y2, y2b)

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    boxAArea = (x2 - x1) * (y2 - y1)
    boxBArea = (x2b - x1b) * (y2b - y1b)
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def run_full_report(snapshot_path: str):
    """
    Call detect_and_map.py in collision mode on snapshot_path.
    This will:
      - run full analysis
      - send full collision report with image to Firebase /collisions
    """
    DETECT_SCRIPT = "/home/hassanm/smart-parking-edge/detect_and_map.py"
    python_bin = sys.executable  # same venv python

    if not os.path.exists(DETECT_SCRIPT):
        print(f"[collision] detect_and_map.py not found at {DETECT_SCRIPT}")
        return

    try:
        subprocess.Popen(
            [
                python_bin,
                DETECT_SCRIPT,
                "--mode", "collision",
                "--image", snapshot_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[collision] spawned detect_and_map.py for full report on {snapshot_path}")
    except Exception as e:
        print("[collision] failed to spawn detect_and_map.py:", e)


# ------------- MAIN LOOP -------------

def main():
    print("[collision] Loading YOLO model:", YOLO_MODEL)
    model = YOLO(YOLO_MODEL)

    last_collision_ts = 0.0

    while True:
        img = fetch_image(IMAGE_URL)
        if img is None:
            print("[collision] no image from camera, retrying...")
            time.sleep(2)
            continue

        h, w = img.shape[:2]

        try:
            res = model(img, verbose=False)[0]
        except Exception as e:
            print("[collision] YOLO error:", e)
            time.sleep(1)
            continue

        names = res.names
        car_boxes = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id)).lower()
            if cls_name in {"car", "truck", "bus", "motorbike", "motorcycle"}:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                car_boxes.append((x1, y1, x2, y2))

        if len(car_boxes) < 2:
            time.sleep(0.8)
            continue

        # check pairwise IoU
        collision_now = False
        max_iou = 0.0
        for i in range(len(car_boxes)):
            for j in range(i + 1, len(car_boxes)):
                iou_val = iou(car_boxes[i], car_boxes[j])
                if iou_val > max_iou:
                    max_iou = iou_val
                if iou_val >= COLLISION_IOU_THRESH:
                    collision_now = True
                    break
            if collision_now:
                break

        if collision_now:
            now = time.time()
            if now - last_collision_ts >= COLLISION_COOLDOWN:
                last_collision_ts = now
                ts_ms = int(now * 1000)
                fname = os.path.join(
                    COLLISION_IMG_DIR,
                    f"collision_raw_{ts_ms}.jpg"
                )
                cv2.imwrite(fname, img)
                print(f"[collision] COLLISION detected (IoU={max_iou:.2f}) -> snapshot saved: {fname}")
                run_full_report(fname)
            else:
                print(f"[collision] collision detected but in cooldown (IoU={max_iou:.2f})")
        else:
            # no collision in this frame
            pass

        time.sleep(0.8)


if __name__ == "__main__":
    print("⏳ Collision watcher running… Ctrl+C to stop")
    main()
