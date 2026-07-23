# check_alignment.py
# (live AI monitor + alignment tool)

import cv2
import numpy as np
import requests
import json
import time
import sys

# === CONSTANTS ===
IMAGE_URL = ""
SPOTS_JSON_FILE = ""

COLOR_SPOT = (0, 255, 0)      # Green
COLOR_ILLEGAL = (0, 0, 255)   # Red
COLOR_COLLISION = (0, 165, 255)  # Orange
COLOR_INFO = (255, 0, 0)      # 🔵 Blue text

THICK = 2


# -------------------------- HELPERS --------------------------

def fetch_image(url: str):
    """Fetch frame from Pi camera server."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error fetching image: {e}", file=sys.stderr)
        return None


def load_spots(path: str):
    """Load spots.json as dictionary spot_key → [x1,y1,x2,y2]."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading spots.json: {e}", file=sys.stderr)
        return {}


def box_iou(b1, b2):
    """Compute IoU between two boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    box1Area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    box2Area = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return interArea / float(box1Area + box2Area - interArea)


# -------------------------- MAIN --------------------------

def main():
    spots = load_spots(SPOTS_JSON_FILE)
    if not spots:
        print("❌ Could not load parking spots.")
        return

    print("▶️ AI Monitor + Alignment started — Press 'q' to quit")

    while True:
        frame = fetch_image(IMAGE_URL)
        if frame is None:
            time.sleep(1)
            continue

        H, W = frame.shape[:2]

        # ---------------- DRAW PARKING SPOTS ----------------
        for key, (x1, y1, x2, y2) in spots.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SPOT, THICK)
            cv2.putText(frame, key, (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SPOT, 2)

        # ---------------- AI SIMULATION ----------------
        # ⚠️ لاحقاً ستربط هذا مع YOLO من detect_and_map.py
        # أما الآن فمجرد placeholder ليتغير لاحقاً
        illegal_count = 0
        collision_count = 0
        occupied = 0

        # Example: show zero values (will be real later)
        info_text = f"Illegal cars: {illegal_count} | Collisions: {collision_count} | Occupied: {occupied}"

        # ---------------- DRAW BLUE INFO TEXT (BOTTOM) ----------------
        baseline_y = H - 20
        cv2.putText(frame,
                    info_text,
                    (20, baseline_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_INFO, 2)

        # ---------------- SHOW FRAME ----------------
        cv2.imshow("AI Monitor + Alignment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("▶️ Alignment monitor stopped.")


if __name__ == "__main__":
    main()

