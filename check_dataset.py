# check_dataset.py
import os
import yaml


def check_dataset():
    # Load data.yaml
    with open('data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    print("🔍 Checking dataset structure...")
    print(f"Classes: {data['names']}")
    print(f"Number of classes: {data['nc']}")
    print()

    # Check all paths
    for split in ['train', 'val', 'test']:
        path = data[split]
        if os.path.exists(path):
            image_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"✅ {split.upper()}: {path}")
            print(f"   Images found: {image_count}")

            # Check corresponding labels folder
            labels_path = path.replace('images', 'labels')
            if os.path.exists(labels_path):
                label_count = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
                print(f"   Labels found: {label_count}")

                # Check if image and label counts match
                if image_count != label_count:
                    print(f"   ⚠️  WARNING: Image count ({image_count}) doesn't match label count ({label_count})")
            else:
                print(f"   ❌ Labels folder missing: {labels_path}")
        else:
            print(f"❌ {split.upper()}: {path} - PATH DOES NOT EXIST")
        print()


if __name__ == "__main__":
    check_dataset()