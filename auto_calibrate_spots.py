# auto_calibrate_spots.py
# هدف هذا السكريبت: جلب الصورة من Pi، اكتشاف المواقف تلقائياً باستخدام OpenCV، 
# التحقق من التداخل، وكتابة ملف spots.json بالتنسيق الصحيح (قاموس).

import cv2
import numpy as np
import requests
import json
import os
import sys
from typing import List, Dict, Tuple

# === الثوابت والإعدادات ===
IMAGE_URL = ""
SPOTS_JSON_FILE = "spots.json"

# معايير الرؤية الحاسوبية (تحتاج إلى ضبط دقيق)
HOUGH_THRESHOLD = 80
MAX_OVERLAP_IOU = 0.05 
SCALE_PERCENT = 100 # لتصغير الصورة (50% من الحجم الأصلي) وتسريع المعالجة

# === الدوال المساعدة ===

def fetch_image(url: str):
    """جلب الصورة من Raspberry Pi عبر HTTP."""
    print(f"Fetching image from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
             print("Error: Could not decode image content.")
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from Pi: {e}", file=sys.stderr)
        return None

def io_u(box_a: List[int], box_b: List[int]) -> float:
    """تحسب IoU لصندوقين [x_min, y_min, x_max, y_max]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    if union_area == 0: return 0.0
    return inter_area / union_area

def cluster_lines(lines, axis_index, distance_threshold=20):
    """تجميع الخطوط المتقاربة بناءً على إحداثي واحد."""
    if not lines: return []
    lines.sort(key=lambda x: x[axis_index])
    
    clusters = []
    current_cluster = [lines[0]]
    
    for i in range(1, len(lines)):
        if lines[i][axis_index] - current_cluster[-1][axis_index] < distance_threshold:
            current_cluster.append(lines[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [lines[i]]
    clusters.append(current_cluster)
    
    return [np.median([l[axis_index] for l in c]) for c in clusters]


# =================================================================
# الدالة الرئيسية للاكتشاف
# =================================================================

def detect_and_define_spots(image) -> List[List[int]]:
    """
    تستخدم OpenCV لاكتشاف الخطوط المستقيمة وتجميعها لتحديد إحداثيات المواقف.
    الناتج: قائمة بصناديق المواقف المكتشفة بصيغة [x_min, y_min, x_max, y_max].
    """
    
    # === 1. المعالجة المسبقة واكتشاف الحواف ===
    width = int(image.shape[1] * SCALE_PERCENT / 100)
    height = int(image.shape[0] * SCALE_PERCENT / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # === 2. اكتشاف الخطوط ===
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=HOUGH_THRESHOLD, 
        minLineLength=50, 
        maxLineGap=10
    )
    
    if lines is None:
        return []

    # === 3. تصنيف وتجميع الخطوط ===
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan2(y2 - y1, x2 - x1)
        else:
            angle = np.pi / 2
        
        angle_deg = np.degrees(angle)
        
        if (abs(angle_deg) < 10) or (abs(angle_deg) > 170):
            horizontal_lines.append((min(y1, y2), x1, x2))
        elif (abs(angle_deg) > 80) and (abs(angle_deg) < 100):
            vertical_lines.append((min(x1, x2), y1, y2))

    distinct_y = cluster_lines(horizontal_lines, 0)
    distinct_x = cluster_lines(vertical_lines, 0)
    
    # 4. تحديد المواقف عبر التقاطع
    parking_spots = []
    scale_factor = 100 / SCALE_PERCENT 
    
    if len(distinct_x) >= 2 and len(distinct_y) >= 2:
        for i in range(len(distinct_x) - 1):
            for j in range(len(distinct_y) - 1):
                x_min = int(distinct_x[i] * scale_factor)
                x_max = int(distinct_x[i+1] * scale_factor)
                y_min = int(distinct_y[j] * scale_factor)
                y_max = int(distinct_y[j+1] * scale_factor)
                
                if (x_max - x_min > 50) and (y_max - y_min > 50):
                    parking_spots.append([x_min, y_min, x_max, y_max])
    
    if not parking_spots:
        print("Could not form coherent parking spots from detected lines.")
        
    return parking_spots


# =================================================================
# دالة الحفظ والتحقق من التداخل (تم التعديل لإنشاء القاموس الصحيح)
# =================================================================

def process_and_save_spots(raw_spots: List[List[int]]):
    """تتحقق من التداخل وتكتب المواقف النظيفة إلى ملف JSON بصيغة قاموس."""
    clean_spots = {} # ⚠️ التعديل هنا: قاموس وليس قائمة
    spot_counter = 1
    
    for i in range(len(raw_spots)):
        spot_a = raw_spots[i]
        is_clean = True
        
        # التحقق من التداخل
        for j in range(i + 1, len(raw_spots)):
            spot_b = raw_spots[j]
            overlap = io_u(spot_a, spot_b)
            
            if overlap > MAX_OVERLAP_IOU:
                print(f"Warning: Spot {i+1} overlaps with Spot {j+1} by {overlap:.2f} (Skipping Spot {i+1})", file=sys.stderr)
                is_clean = False
                break
        
        if is_clean:
            # كتابة المفتاح والقيمة مباشرة في القاموس
            clean_spots[f"spot_{spot_counter}"] = spot_a 
            spot_counter += 1

    if clean_spots:
        with open(SPOTS_JSON_FILE, 'w') as f:
            # كتابة القاموس مباشرة (التنسيق المطلوب لـ detect_and_map.py)
            json.dump(clean_spots, f, indent=4) 
        print(f"\nSUCCESS: Wrote {len(clean_spots)} clean spots to {SPOTS_JSON_FILE}")
    else:
        print("\nFAILURE: Could not detect any clean parking spots.", file=sys.stderr)

# =================================================================
# الدالة الرئيسية للتشغيل
# =================================================================

def main():
    print(f"Starting automatic spot calibration...")
    
    # 1. جلب الصورة
    image = fetch_image(IMAGE_URL)
    if image is None:
        print("Calibration failed: Could not retrieve image.", file=sys.stderr)
        return

    # 2. اكتشاف وتحديد الإحداثيات
    raw_spots = detect_and_define_spots(image)
    
    if not raw_spots:
        print("Calibration failed: No parking spots detected.", file=sys.stderr)
        return

    # 3. التحقق من التداخل وحفظ الملف
    process_and_save_spots(raw_spots)

if __name__ == "__main__":
    main()
