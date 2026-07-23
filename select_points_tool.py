# select_points_tool.py
# أداة لتحديد أربع نقاط بالماوس لغرض Perspective Transformation

import cv2
import numpy as np
import requests
import time
import sys
from typing import List

# === الثوابت ===
IMAGE_URL = ""
POINTS = [] # قائمة لتخزين النقاط التي ينقر عليها المستخدم

# === الدوال المساعدة ===

def fetch_image(url: str):
    """جلب الصورة من Raspberry Pi عبر HTTP."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}", file=sys.stderr)
        return None

def mouse_callback(event, x, y, flags, param):
    """وظيفة تُستدعى عند النقر بالماوس."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(POINTS) < 4:
            # تسجيل النقطة الجديدة
            POINTS.append((x, y))
            print(f"Point {len(POINTS)} registered: ({x}, {y})")
            
            # رسم دائرة صغيرة على النقطة
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1) 
            cv2.imshow("Select Points Tool", param)
            
            if len(POINTS) == 4:
                # عند اكتمال 4 نقاط، يتم طباعة المصفوفة النهائية
                print("\n✅ Final SOURCE_POINTS Array:")
                print("====================================")
                print(f"SOURCE_POINTS = np.float32({POINTS})")
                print("====================================")
                print("Press 'q' in the window to quit.")

def main_selection_tool():
    print("Starting point selection tool. Click 4 points on the real-world rectangle.")
    
    # 1. جلب الصورة الأصلية
    original_frame = fetch_image(IMAGE_URL)
    if original_frame is None:
        print("Failed to fetch image. Exiting.")
        return

    # 2. إعداد النافذة ووظيفة الماوس
    cv2.namedWindow("Select Points Tool")
    cv2.setMouseCallback("Select Points Tool", mouse_callback, original_frame)
    
    # 3. عرض الصورة والانتظار
    cv2.imshow("Select Points Tool", original_frame)
    
    # الانتظار حتى يتم النقر على 'q'
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Tool closed.")

if __name__ == '__main__':
    # تأكد من أن المكتبات موجودة (خاصة OpenCV)
    if 'sys' not in locals() and 'sys' not in globals():
        import sys
        
    main_selection_tool()
