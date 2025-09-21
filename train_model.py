# train_road_defect_model.py
import os
from ultralytics import YOLO
import yaml
from datetime import datetime


def train_road_defect_model():
    print("🚀 STARTING YOLOv8 ROAD DEFECT TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"YOLOv8 Version: 8.3.189")
    print(f"Device: CPU")
    print("=" * 60)

    # Load dataset configuration
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"  Dataset Info:")
    print(f"   Classes: {data_config['names']}")
    print(f"   Number of classes: {data_config['nc']}")
    print(f"   Training images: {len(os.listdir('Dataset/train/images'))}")
    print(f"   Validation images: {len(os.listdir('Dataset/val/images'))}")
    print(f"   Test images: {len(os.listdir('Dataset/test/images'))}")
    print("=" * 60)

    # Load YOLOv8 Nano model (best for CPU)
    print("📦 Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')

    # Training parameters optimized for CPU
    training_params = {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'patience': 30,
        'device': 'cpu',
        'workers': 2,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'name': 'road_defect_v1',
        'save': True,
        'save_period': 25,
        'verbose': True,
        'amp': False,  # Disable mixed precision for CPU
        'exist_ok': True  # Overwrite existing runs
    }

    print("⚙️ Training Configuration:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")

    print("\n🎯 Starting training process...")
    print("⏰ This will take several hours on CPU")
    print("💡 You can monitor progress in the 'runs' folder")
    print("=" * 60)

    # Start training
    results = model.train(**training_params)

    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Validation results
    print("📊 Validation Results:")
    print(f"   mAP50-95: {results.box.map:.3f}")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   Precision: {results.box.mp:.3f}")
    print(f"   Recall: {results.box.mr:.3f}")

    # Save final model
    model.save('road_defect_final_model.pt')
    print("💾 Model saved as 'road_defect_final_model.pt'")

    print("=" * 60)
    print("🎉 Training complete! Model is ready for inference.")

    return results


if __name__ == "__main__":
    train_road_defect_model()