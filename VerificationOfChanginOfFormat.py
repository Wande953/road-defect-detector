from pathlib import Path


def verify_yolo_format(dataset_path):
    dataset_path = Path(dataset_path)
    print("Verifying YOLO format...")

    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        labels_path = split_path / 'labels'

        if labels_path.exists():
            label_files = list(labels_path.glob('*.txt'))
            print(f"\n--- {split.upper()} ---")

            # Check first few files
            for label_file in label_files[:3]:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                print(f"{label_file.name}:")
                for i, line in enumerate(lines[:2]):  # Show first 2 lines
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            print(f"  Line {i + 1}: ✅ YOLO format - {line}")
                        else:
                            print(f"  Line {i + 1}: ❌ Wrong format - {line}")


# Verify the conversion
dataset_path = r"C:\Users\202234042\Documents\Dataset"
verify_yolo_format(dataset_path)