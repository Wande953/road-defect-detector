from pathlib import Path
import yaml


def verify_data_yaml():
    dataset_path = Path(r"C:\Users\202234042\Documents\Dataset")
    yaml_file = dataset_path / 'data.yaml'

    if yaml_file.exists():
        print("✅ data.yaml found!")

        # Read and display content
        with open(yaml_file, 'r') as f:
            content = f.read()
        print("File content:")
        print(content)

        # Parse YAML
        try:
            data = yaml.safe_load(content)
            print("\nParsed YAML:")
            for key, value in data.items():
                print(f"  {key}: {value}")

            # Verify paths
            print("\nPath verification:")
            for split in ['train', 'val']:
                if split in data:
                    path_str = data[split]
                    print(f"  {split} path: {path_str}")

                    # Handle relative paths
                    if path_str.startswith('../'):
                        # Remove the '../' and build absolute path
                        relative_path = path_str[3:]  # Remove '../'
                        abs_path = dataset_path.parent / relative_path
                    else:
                        abs_path = Path(path_str)

                    if abs_path.exists():
                        print(f"    ✅ EXISTS: {abs_path}")
                    else:
                        print(f"    ❌ MISSING: {abs_path}")

            return True

        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error: {e}")
            return False
    else:
        print("❌ data.yaml still not found!")
        return False


# Verify the data.yaml file
verify_data_yaml()