import os
from pathlib import Path


def check_dataset_structure(dataset_path):
    dataset_path = Path(dataset_path)
    print(f"Checking dataset structure at: {dataset_path}")

    # Check if main directories exist
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        print(f"\n--- {split.upper()} ---")

        if split_path.exists():
            # Check images
            images_path = split_path / 'images'
            if images_path.exists():
                image_files = list(images_path.glob('*.*'))
                print(f"Images: {len(image_files)} files found")
                if image_files:
                    print(f"  Example: {image_files[0].name}")
            else:
                print("❌ No 'images' folder found")

            # Check labels
            labels_path = split_path / 'labels'
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                print(f"Labels: {len(label_files)} files found")
                if label_files:
                    # Check label content
                    sample_label = label_files[0]
                    with open(sample_label, 'r') as f:
                        content = f.read().strip()
                    print(f"  Example label content: '{content}'")
            else:
                print("❌ No 'labels' folder found")
        else:
            print(f"❌ No '{split}' folder found")


# Run the check
dataset_path = r"C:\Users\202234042\Documents\Dataset"
check_dataset_structure(dataset_path)