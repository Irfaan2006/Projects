"""
=============================================================================
Fine-Tuning Script: YOLOv11 for Custom Wildlife Monitoring & Conservation
=============================================================================
This standalone script allows you to fine-tune pre-trained YOLOv11 models 
(yolo11n, yolo11s, yolo11m) on your custom wildlife camera-trap datasets.

Usage:
    python train_yolov11_wildlife.py --data dataset.yaml --epochs 50 --batch 16 --imgsz 640

Requirements:
    - Custom dataset in YOLO format with data.yaml specifying train/val image paths.
"""

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on Wildlife Dataset")
    parser.add_argument("--data", type=str, default="wildlife_dataset.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained YOLOv11 weight (yolo11n.pt / yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per step")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution input size")
    parser.add_argument("--device", type=str, default="0", help="GPU device index or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train_wildlife", help="Save directory")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print("      INTELLIGENT WILDLIFE MONITORING - YOLOv11 FINE-TUNING      ")
    print("=" * 70)
    print(f"[*] Base Model       : {args.model}")
    print(f"[*] Dataset Config   : {args.data}")
    print(f"[*] Target Epochs    : {args.epochs}")
    print(f"[*] Batch Size       : {args.batch}")
    print(f"[*] Image Resolution : {args.imgsz}x{args.imgsz}")
    print(f"[*] Compute Device   : {args.device}")
    print("-" * 70)

    try:
        from ultralytics import YOLO
        
        # Initialize YOLOv11 pretrained checkpoint
        model = YOLO(args.model)

        # High-Accuracy Hyperparameters tuned for dense foliage and camouflaged animals:
        # - mosaic: 1.0 (synthesizes 4 images into one, excellent for small animals)
        # - mixup: 0.15 (handles overlapping herds)
        # - fliplr: 0.5 (horizontal flip invariance)
        # - hsv_h / hsv_s / hsv_v: Augmentations for variable infrared / night vision lighting
        print("[*] Launching YOLOv11 Training Loop with High-Accuracy Hyperparameters...")
        
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name="yolo11_wildlife_exp",
            mosaic=1.0,
            mixup=0.15,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            save=True,
            val=True,
            plots=True
        )

        print("[+] Training Complete!")
        print(f"[+] Best Trained Weights saved to: {os.path.join(args.project, 'yolo11_wildlife_exp', 'weights', 'best.pt')}")

        # Optional ONNX export for edge deployment (Nvidia Jetson / Raspberry Pi / Flask production)
        print("[*] Exporting model to ONNX format for accelerated inference...")
        model.export(format="onnx", imgsz=args.imgsz, dynamic=True)
        print("[+] ONNX Export Successful!")

    except ImportError:
        print("[!] Ultralytics library not detected. Install via: pip install ultralytics")
    except Exception as e:
        print(f"[!] Training error occurred: {e}")

if __name__ == "__main__":
    main()
