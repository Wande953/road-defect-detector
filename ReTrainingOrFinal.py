from ultralytics import YOLO
import torch
from pathlib import Path


def train_pothole_detector():
    print("🚀 Starting YOLO Training with 8 Classes")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Display class information
    dataset_path = Path(r"C:\Users\202234042\Documents\Dataset")
    yaml_file = dataset_path / 'data.yaml'

    import yaml
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    print(f"\n📊 Dataset Info:")
    print(f"  Classes: {data['nc']}")
    print(f"  Class names: {data['names']}")
    print(f"  Train images: {len(list((dataset_path / 'train' / 'images').glob('*.*')))}")
    print(f"  Val images: {len(list((dataset_path / 'val' / 'images').glob('*.*')))}")
    print(f"  Test images: {len(list((dataset_path / 'test' / 'images').glob('*.*')))}")

    # Initialize model - using yolov5n (you can choose larger models if needed)
    model = YOLO('yolov5n.pt')  # Options: 'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', etc.

    # Train the model
    print("\n🎯 Starting training...")
    results = model.train(
        data=str(yaml_file),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=30,  # Early stopping patience
        name='8class_drain_pothole_detection',
        optimizer='Adam',  # 'Adam', 'AdamW', 'SGD'
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        verbose=True,  # Show detailed output
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
    )

    print("✅ Training completed successfully!")
    print(f"📁 Results saved to: {results.save_dir}")

    return results


def test_trained_model():
    """Test the trained model on sample images"""
    print("\n🧪 Testing trained model...")

    # Load the best model
    model = YOLO('runs/detect/8class_drain_pothole_detection/weights/best.pt')

    # Test on a few images from each set
    test_images = [
        r"C:\Users\202234042\Documents\Dataset\val\images\000024_r_jpg.rf.63ee82999c57dfa695a8eb89cc605322.jpg",
        r"C:\Users\202234042\Documents\Dataset\test\images\0000060_jpg.rf.edcd7edac419b127fea4793fae054732.jpg",
        r"C:\Users\202234042\Documents\Dataset\train\images\0000060_jpg.rf.fc50c4df0a7ec8bb294fb73e015d276c.jpg"
    ]

    for img_path in test_images:
        img_path = Path(img_path)
        if img_path.exists():
            print(f"\n🔍 Testing: {img_path.name}")
            results = model(str(img_path))

            print(f"   Detected {len(results[0].boxes)} objects:")
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                print(
                    f"     Object {i + 1}: Class {class_id} ({data['names'][class_id]}) - Confidence: {confidence:.3f}")

            # Save results
            results[0].save(f"result_{img_path.name}")
            print(f"   ✅ Result saved: result_{img_path.name}")
        else:
            print(f"❌ Image not found: {img_path}")


if __name__ == "__main__":
    # Train the model
    results = train_pothole_detector()

    # Test the trained model
    test_trained_model()

    print("\n🎉 All done! Your model is ready for deployment!")