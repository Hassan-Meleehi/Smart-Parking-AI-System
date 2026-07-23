README
🅿️ Smart Parking AI System
An end-to-end Smart Parking system that combines Artificial Intelligence, IoT (Edge Computing), and Cloud Computing to automate parking space monitoring, detect violations, and give facility managers a real-time dashboard.

Computer Engineering Graduation Project — Jazan University, College of Computer Science and Information Technology, Department of Computer Engineering & Networks.

📌 Overview
Traditional parking lots rely on manual patrols and human oversight, causing wasted time, fuel, and inconsistent enforcement. This project builds an autonomous pipeline that:
Captures live footage from an HD camera mounted above the parking area
Runs YOLOv8 on a Raspberry Pi 4 at the edge to detect vehicles in real time
Maps detected vehicles to parking spot polygons using IoU (Intersection over Union) geometry
Detects violations (wrong-spot parking, unauthorized zone access)
Streams results to Firebase Realtime Database
Renders a live Flutter Web dashboard for administrators

🏗️ System Architecture
[HD Camera] → [Raspberry Pi 4 + YOLOv8 Inference] → [Spot Mapping / Violation Logic]
      → [Firebase Realtime Database] → [Flutter Web Dashboard]


Image Capture
HD USB camera (1080p, 30fps) mounted above the lot

Edge Processing
Raspberry Pi 4 (4GB) runs YOLOv8 inference locally

Cloud Sync
Firebase Realtime Database receives JSON detection + violation payloads

Visualization
Flutter Web app subscribes to Firebase streams for live updates

🧠 AI Model — YOLOv8
Fine-tuned on a custom dataset of 2,500+ annotated parking lot images
Transfer learning from COCO weights, 100 training epochs with data augmentation
~92% mAP@0.5 IoU, ~15 FPS on Raspberry Pi 4, 6.2MB model size (edge-optimized)
Detects: cars, trucks, motorcycles

Spot occupancy logic
if calculate_iou(vehicle_box, spot_polygon) > 0.4:
    spot.status = "occupied"
    spot.vehicle_id = vehicle.id
else:
    spot.status = "empty"

🚨 Violation Detection
Violation Type         Logic

Wrong-Spot Parking
Vehicle bounding box overlaps multiple spot polygons (>40% + >20% adjacent)

Unauthorized Zone Access
Vehicle centroid falls inside a restricted polygon (fire lanes, handicapped zones, loading docks)

☁️ Firebase Data Structure
Parking Status
{
  "parking_status": {
    "spot_A1": { "id": "A1", "status": "occupied", "vehicle_detected": true, "last_updated": "..." },
    "total_spots": 50,
    "occupied_count": 32,
    "available_count": 18
  }
}

Violations
{
  "violations": {
    "violation_001": { "type": "wrong_spot", "spot_id": "B5", "severity": "medium", "timestamp": "..." }
  }
}

🛠️ Tech Stack
Python 3.9+ — core processing & orchestration
OpenCV — image preprocessing & annotation
YOLOv8 (Ultralytics) — real-time object detection
Firebase Realtime Database — cloud sync backend
Flutter Web — admin dashboard
Raspberry Pi 4 Model B (4GB) — edge compute unit
Git & GitHub — version control

🧩 Known Limitations & Challenges
Firebase free tier (1GB storage / 10GB monthly transfer) limits historical image storage for violation evidence
Collision detection (prototype) struggles with monocular depth perception — stereo vision/LiDAR needed for reliable 3D distance estimation
Network resilience — implemented connection pooling, exponential backoff retry, and local SQLite caching for offline periods

🚀 Future Enhancements
Cloud image storage (Firebase Storage / AWS S3) for a visual violation audit trail
Expand training dataset to 10,000+ images; explore YOLOv9
Native iOS/Android app with push notifications
License plate recognition (OCR) for automated access control & payment integration

🧑‍💻 About This Project
Built independently from the ground up — hardware selection, circuit/wiring setup, model training, backend integration, and dashboard development — as a graduation project under academic supervision.

📄 License
This project is shared for educational and portfolio purposes.

