# train_road_defect_model_yolov5.py
import os
import yaml
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def setup_yolov5_environment():
    """Setup YOLOv5 environment and requirements"""
    print("🔧 Setting up YOLOv5 environment...")

    # Clone YOLOv5 if not exists
    if not os.path.exists("yolov5"):
        print("📦 Cloning YOLOv5 repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"],
                           check=True, capture_output=True)
            print("✅ YOLOv5 cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone YOLOv5: {e}")
            return False

    # Install requirements
    print("📦 Installing YOLOv5 requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", "yolov5/requirements.txt"
        ], check=True, capture_output=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

    return True


def create_yolov5_data_config():
    """Create YOLOv5 compatible data configuration"""
    print("📊 Creating YOLOv5 data configuration...")

    # Load original data config
    with open('data.yaml', 'r') as f:
        original_config = yaml.safe_load(f)

    # Create YOLOv5 compatible config
    yolov5_config = {
        'path': str(Path.cwd()),
        'train': 'Dataset/train/images',
        'val': 'Dataset/val/images',
        'test': 'Dataset/test/images',
        'nc': original_config['nc'],
        'names': original_config['names']
    }

    # Save YOLOv5 config
    with open('data_yolov5.yaml', 'w') as f:
        yaml.dump(yolov5_config, f)

    print("✅ YOLOv5 data config created: data_yolov5.yaml")
    return 'data_yolov5.yaml'


def train_yolov5_road_defect():
    print("🚀 STARTING YOLOv5 ROAD DEFECT TRAINING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: CPU")
    print("=" * 60)

    # Setup environment
    if not setup_yolov5_environment():
        print("❌ Failed to setup YOLOv5 environment")
        return None

    # Create YOLOv5 data config
    data_config = create_yolov5_data_config()

    # Load dataset configuration
    with open('data.yaml', 'r') as f:
        data_config_orig = yaml.safe_load(f)

    print(f"📊 Dataset Info:")
    print(f"   Classes: {data_config_orig['names']}")
    print(f"   Number of classes: {data_config_orig['nc']}")
    print(f"   Training images: {len(os.listdir('Dataset/train/images'))}")
    print(f"   Validation images: {len(os.listdir('Dataset/val/images'))}")
    print(f"   Test images: {len(os.listdir('Dataset/test/images'))}")
    print("=" * 60)

    # YOLOv5 training command parameters
    training_params = [
        "python", "train.py",
        "--data", data_config,
        "--weights", "yolov5s.pt",
        "--epochs", "100",
        "--img", "640",
        "--batch", "8",
        "--patience", "30",
        "--device", "cpu",
        "--workers", "2",
        "--optimizer", "Adam",
        "--lr0", "0.001",
        "--lrf", "0.01",
        "--name", "road_defect_v1_yolov5",
        "--project", "training_results_yolov5",
        "--exist-ok",
        "--save-period", "25",
        "--verbose"
    ]

    print("⚙️ YOLOv5 Training Configuration:")
    print(f"   Model: yolov5s.pt")
    print(f"   Epochs: 100")
    print(f"   Image size: 640")
    print(f"   Batch size: 8")
    print(f"   Device: CPU")
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: 0.001")
    print(f"   Project: training_results_yolov5")

    print("\n🎯 Starting YOLOv5 training process...")
    print("⏰ This will take several hours on CPU")
    print("💡 You can monitor progress in the 'training_results_yolov5' folder")
    print("=" * 60)

    # Start YOLOv5 training
    try:
        # Change to yolov5 directory
        original_dir = os.getcwd()
        os.chdir("yolov5")

        print("🏃‍♂️ Running YOLOv5 training command...")
        print(" ".join(training_params))

        # Run training
        result = subprocess.run(
            training_params,
            capture_output=True,
            text=True,
            timeout=None  # No timeout for long training
        )

        # Print output
        print("\n📋 Training Output:")
        print(result.stdout)

        if result.stderr:
            print("❌ Errors:")
            print(result.stderr)

        # Return to original directory
        os.chdir(original_dir)

        if result.returncode == 0:
            print("✅ YOLOv5 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Check results
            results_dir = "training_results_yolov5/road_defect_v1_yolov5"
            if os.path.exists(results_dir):
                print("📊 Training results available in:")
                print(f"   {results_dir}")
                print("   - results.png (training curves)")
                print("   - weights/best.pt (best model)")
                print("   - weights/last.pt (last model)")

            print("=" * 60)
            print("🎉 YOLOv5 Training complete! Model is ready for inference.")

            return True

        else:
            print("❌ YOLOv5 training failed")
            return False

    except Exception as e:
        print(f"❌ Training error: {e}")
        # Return to original directory in case of error
        os.chdir(original_dir)
        return False


def validate_yolov5_model():
    """Validate the trained YOLOv5 model"""
    print("\n🔍 Validating YOLOv5 model...")

    val_command = [
        "python", "val.py",
        "--data", "../data_yolov5.yaml",
        "--weights", "training_results_yolov5/road_defect_v1_yolov5/weights/best.pt",
        "--img", "640",
        "--conf", "0.25",
        "--device", "cpu",
        "--verbose"
    ]

    try:
        os.chdir("yolov5")
        result = subprocess.run(val_command, capture_output=True, text=True)
        print(result.stdout)

        if result.stderr:
            print("Validation errors:", result.stderr)

    except Exception as e:
        print(f"Validation error: {e}")
    finally:
        os.chdir("..")


if __name__ == "__main__":
    success = train_yolov5_road_defect()

    if success:
        # Optional: Run validation
        validate_yolov5_model()
    else:
        print("❌ Training failed. Please check the error messages.")