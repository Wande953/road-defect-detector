from pathlib import Path
import cv2
import numpy as np


def convert_polygon_to_yolo(dataset_path):
    dataset_path = Path(dataset_path)
    print("Converting polygon labels to YOLO format...")

    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        labels_path = split_path / 'labels'
        images_path = split_path / 'images'

        if labels_path.exists() and images_path.exists():
            label_files = list(labels_path.glob('*.txt'))
            print(f"\n--- Converting {split} labels ({len(label_files)} files) ---")

            converted_count = 0
            empty_count = 0

            for label_file in label_files:
                # Find corresponding image to get dimensions
                image_file = images_path / label_file.name.replace('.txt', '.jpg')
                if not image_file.exists():
                    # Try other extensions
                    for ext in ['.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        image_file = images_path / label_file.name.replace('.txt', ext)
                        if image_file.exists():
                            break

                if image_file.exists():
                    # Read image to get dimensions
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        img_height, img_width = image.shape[:2]

                        # Read original label content
                        with open(label_file, 'r') as f:
                            lines = f.readlines()

                        new_lines = []
                        for line in lines:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 8:  # Polygon format
                                    try:
                                        # Extract polygon coordinates
                                        coords = [float(parts[i]) for i in range(8)]
                                        class_name = parts[8] if len(parts) > 8 else 'pothole'

                                        # Convert polygon to bounding box
                                        x_coords = coords[0::2]  # x1, x2, x3, x4
                                        y_coords = coords[1::2]  # y1, y2, y3, y4

                                        x_min, x_max = min(x_coords), max(x_coords)
                                        y_min, y_max = min(y_coords), max(y_coords)

                                        # Calculate YOLO format (normalized)
                                        center_x = (x_min + x_max) / 2 / img_width
                                        center_y = (y_min + y_max) / 2 / img_height
                                        width = (x_max - x_min) / img_width
                                        height = (y_max - y_min) / img_height

                                        # Class ID (assuming only 'pothole' class)
                                        class_id = 0  # Change if you have multiple classes

                                        # Add to new lines
                                        new_lines.append(
                                            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                                        converted_count += 1

                                    except (ValueError, IndexError) as e:
                                        print(f"Error processing {label_file.name}: {e}")
                                        continue

                        # Write converted labels back to file
                        with open(label_file, 'w') as f:
                            f.write('\n'.join(new_lines))

                        if not new_lines:
                            empty_count += 1

            print(f"Converted {converted_count} polygons to bounding boxes")
            print(f"Empty files: {empty_count}")

        else:
            print(f"❌ Missing folders in {split}")


# Convert the labels
dataset_path = r"C:\Users\202234042\Documents\Dataset"
convert_polygon_to_yolo(dataset_path)