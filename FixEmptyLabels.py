# convert_labels_to_yolo.py
import os
import yaml


def convert_labels_to_yolo_format():
    print("🔄 Converting Labels to YOLO Format")
    print("=" * 50)

    # Load class names from data.yaml
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"Class names: {class_names}")

    splits = ['train', 'val', 'test']

    for split in splits:
        label_dir = f"Dataset/{split}/labels"
        converted_count = 0
        error_count = 0

        print(f"\n📁 Processing {split.upper()} labels...")

        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(label_dir, label_file)

                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()

                    if not content:
                        continue  # Skip empty files

                    lines = content.split('\n')
                    new_lines = []

                    for line in lines:
                        parts = line.split()

                        if len(parts) == 10:
                            # This is the problematic format
                            # Extract class name (usually at position 8 or 9)
                            class_name = parts[8]  # Try position 8

                            # If it's a number, try position 9
                            if class_name.isdigit():
                                class_name = parts[9] if len(parts) > 9 else "Unknown"

                            # Find class ID
                            if class_name in class_names:
                                class_id = class_names.index(class_name)

                                # For now, create a placeholder bounding box
                                # You'll need proper coordinates from your data
                                x_center = 0.5  # Center of image
                                y_center = 0.5  # Center of image
                                width = 0.3  # 30% of image width
                                height = 0.3  # 30% of image height

                                new_line = f"{class_id} {x_center} {y_center} {width} {height}"
                                new_lines.append(new_line)
                                converted_count += 1
                            else:
                                print(f"   ⚠️ Unknown class: {class_name} in {label_file}")
                                error_count += 1

                        elif len(parts) == 5:
                            # Already in YOLO format, keep it
                            new_lines.append(line)

                        else:
                            print(f"   ⚠️ Unexpected format in {label_file}: {len(parts)} values")
                            error_count += 1

                    # Write converted labels back to file
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(new_lines))

                except Exception as e:
                    print(f"   ❌ Error processing {label_file}: {e}")
                    error_count += 1

        print(f"   ✅ Converted: {converted_count} labels")
        print(f"   ❌ Errors: {error_count} labels")

    print("\n🎯 Conversion completed!")
    print("💡 Note: This created placeholder bounding boxes.")
    print("   You may need to properly annotate your images for accurate results.")


if __name__ == "__main__":
    convert_labels_to_yolo_format()