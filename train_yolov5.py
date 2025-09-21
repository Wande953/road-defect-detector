# train_yolov5.py
import os
import torch
import yaml
from pathlib import Path
import argparse
import subprocess
import sys
from datetime import datetime


class YOLOv5Trainer:
    def __init__(self, data_dir="Dataset", project_name="road_defect_detection"):
        self.data_dir = data_dir
        self.project_name = project_name
        self.yolov5_dir = "yolov5"  # Path to yolov5 directory

        # Check if yolov5 is available
        if not os.path.exists(self.yolov5_dir):
            print("YOLOv5 not found. Please clone it from: https://github.com/ultralytics/yolov5")
            sys.exit(1)

        print("YOLOv5 Trainer Initialized")

    def create_yaml_config(self):
        """Create YAML configuration file for YOLOv5 training"""
        config = {
            'path': os.path.abspath(self.data_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'Alligator',
                1: 'Longitudinal',
                2: 'Pothole',
                3: 'Transverse'
            },
            'nc': 4
        }

        config_path = os.path.join(self.data_dir, 'road_defect.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        print(f"YAML configuration created: {config_path}")
        return config_path

    def prepare_dataset_structure(self):
        """Ensure dataset is in YOLOv5 format"""
        # YOLOv5 expects specific directory structure
        for split in ['train', 'val', 'test']:
            # Create labels directories if they don't exist
            labels_dir = os.path.join(self.data_dir, 'labels', split)
            os.makedirs(labels_dir, exist_ok=True)

        print("Dataset structure prepared for YOLOv5")

    def train(self, epochs=50, batch_size=16, img_size=640):
        """Train YOLOv5 model"""
        config_path = self.create_yaml_config()
        self.prepare_dataset_structure()

        # Set up training command
        cmd = [
            'python', os.path.join(self.yolov5_dir, 'train.py'),
            '--img', str(img_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--data', config_path,
            '--weights', 'yolov5s.pt',  # Using small model for faster training
            '--project', self.project_name,
            '--name', f'yolov5_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        ]

        print(f"Starting YOLOv5 training with command: {' '.join(cmd)}")

        # Run training
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("YOLOv5 training completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"YOLOv5 training failed: {e}")
            print(f"Error output: {e.stderr}")
            return False

        return True

    def export_results(self, run_dir):
        """Export training results for comparison"""
        results = {}

        # Find the best model
        weights_dir = os.path.join(run_dir, 'weights')
        if os.path.exists(weights_dir):
            best_model = os.path.join(weights_dir, 'best.pt')
            if os.path.exists(best_model):
                results['best_model'] = best_model

        # Parse results from CSV or other output files
        results_file = os.path.join(run_dir, 'results.csv')
        if os.path.exists(results_file):
            # Parse CSV results
            import pandas as pd
            df = pd.read_csv(results_file)
            results['metrics'] = df.to_dict()

        return results


def compare_models(yolov5_results, yolov8_results):
    """Compare YOLOv5 and YOLOv8 performance"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: YOLOv5 vs YOLOv8")
    print("=" * 60)

    # This is a simplified comparison - you'd need to extract actual metrics
    # from both training runs

    print("Comparison metrics would include:")
    print("- mAP@0.5 (mean Average Precision)")
    print("- mAP@0.5:0.95")
    print("- Precision")
    print("- Recall")
    print("- F1 Score")
    print("- Training time")
    print("- Inference speed")
    print("- Model size")

    print("\nTo get a proper comparison, you should:")
    print("1. Train both models on the same dataset")
    print("2. Evaluate both on the same test set")
    print("3. Compare metrics like mAP, precision, recall")
    print("4. Compare inference speed on the same hardware")

    return {"yolov5": yolov5_results, "yolov8": yolov8_results}


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 for road defect detection')
    parser.add_argument('--data-dir', default='Dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')

    args = parser.parse_args()

    # Initialize trainer
    trainer = YOLOv5Trainer(data_dir=args.data_dir)

    # Start training
    success = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    if success:
        print("YOLOv5 training completed!")
        # Here you would typically load YOLOv8 results and compare
        # For now, we'll just note that comparison should be done
        print("\nTo compare with YOLOv8:")
        print("1. Train your YOLOv8 model with similar parameters")
        print("2. Evaluate both models on the same test set")
        print("3. Use the compare_models() function with both results")
    else:
        print("YOLOv5 training failed!")


if __name__ == "__main__":
    main()