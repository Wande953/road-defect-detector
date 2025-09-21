from pathlib import Path
import yaml


def create_data_yaml():
    dataset_path = Path(r"C:\Users\202234042\Documents\Dataset")
    yaml_file = dataset_path / 'data.yaml'

    # YAML content structure
    data_config = {
        'train': '../Dataset/train/images',
        'val': '../Dataset/val/images',
        'test': '../Dataset/test/images',
        'nc': 8,  # Number of classes (only pothole)
        'names': ['Drain Hole', 'circle-drain-clean-', 'drain-clean-', 'drain-not clean-', 'hole', 'manhole', 'pothole', 'sewer cover']  # Class names
    }

    # Write the YAML file
    with open(yaml_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"✅ Created data.yaml at: {yaml_file}")
    print("Content:")
    print(yaml.dump(data_config, default_flow_style=False))

    # Verify the file was created
    if yaml_file.exists():
        print("✅ File verification: SUCCESS")
        return True
    else:
        print("❌ File verification: FAILED")
        return False


# Create the data.yaml file
create_data_yaml()