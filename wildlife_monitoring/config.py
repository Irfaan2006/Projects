import os

# Project Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Upload and Output Storage Paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

# Create directories if they do not exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# YOLOv11 Detection Configurations
YOLO_MODEL_NAME = "yolo11n.pt"  # Will automatically download via ultralytics if missing
YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45
USE_TTA = False  # Test-Time Augmentation switch for boosted accuracy on static frames

# Wildlife Target Classes (COCO standard animal indices & names)
# 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
WILDLIFE_CLASSES = {
    14: "Bird",
    15: "Cat / Felid",
    16: "Wolf / Wild Canid",
    17: "Horse / Wild Equid",
    18: "Sheep / Caprid",
    19: "Bovid / Antelope",
    20: "Elephant",
    21: "Bear",
    22: "Zebra",
    23: "Giraffe"
}

# DeepSORT Tracking Configurations
DEEPSORT_MAX_COSINE_DISTANCE = 0.3
DEEPSORT_NN_BUDGET = 100
DEEPSORT_MAX_IOU_DISTANCE = 0.7
DEEPSORT_MAX_AGE = 60       # Frames to keep track without matching detection
DEEPSORT_N_INIT = 3         # Consecutive detections required to confirm track ID

# Danger Zone / Virtual Fence Polygon Default (relative percentages of frame width and height)
# Format: List of (x_ratio, y_ratio)
DANGER_ZONE_POLYGON_DEFAULT = [
    (0.65, 0.10),
    (0.98, 0.10),
    (0.98, 0.90),
    (0.65, 0.90)
]

# Flask App Configuration
SECRET_KEY = "intelligent-wildlife-monitoring-yolov11-deepsort-secret-key"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
