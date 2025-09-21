# test_road_defect_model.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import os
import numpy as np


class RoadDefectTester:
    def __init__(self, confidence_threshold=0.5):
        print("Initializing Road Defect Tester...")
        # Load your NEW trained model
        self.model = YOLO('road_defect_final_model.pt')  # Your new model
        self.conf_threshold = confidence_threshold
        self.class_names = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']  # NEW classes
        print("Model loaded successfully!")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"Detecting {len(self.class_names)} classes: {self.class_names}")

    def _clean_path(self, path_string):
        """Remove quotes and extra spaces from path"""
        return path_string.strip().strip('"').strip("'")

    def detect_defects(self, image_path):
        """Detect road defects in an image"""
        try:
            results = self.model(image_path, conf=self.conf_threshold)
            detections = []

            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()

                detections.append({
                    'class': self.class_names[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'severity': self._get_severity(confidence),
                    'timestamp': datetime.now().isoformat()
                })

            return detections, results[0]
        except Exception as e:
            print(f"Detection error: {e}")
            return [], None

    def _get_severity(self, confidence):
        """Convert confidence to severity level"""
        if confidence > 0.7:
            return "HIGH"
        elif confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _draw_circle_annotations(self, image, detections):
        """Draw circles around detected defects instead of rectangles"""
        annotated_image = image.copy()

        # Define colors for different defect types
        colors = {
            'Alligator': (0, 0, 255),  # Red
            'Longitudinal': (0, 255, 0),  # Green
            'Pothole': (255, 0, 0),  # Blue
            'Transverse': (255, 255, 0)  # Cyan
        }

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Calculate radius based on bounding box size
            width = x2 - x1
            height = y2 - y1
            radius = int(max(width, height) / 2)

            # Get color based on defect type
            color = colors.get(detection['class'], (255, 255, 255))

            # Draw circle
            cv2.circle(annotated_image, (center_x, center_y), radius, color, 3)

            # Draw filled background for text
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_image,
                          (center_x - text_size[0] // 2 - 5, center_y - radius - text_size[1] - 10),
                          (center_x + text_size[0] // 2 + 5, center_y - radius - 5),
                          color, -1)

            # Draw text
            cv2.putText(annotated_image, label,
                        (center_x - text_size[0] // 2, center_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_image

    def process_image(self, image_path, save_result=True):
        """Process single image and return results"""
        # Clean the path
        clean_path = self._clean_path(image_path)
        image_name = Path(clean_path).name

        print(f"Processing: {image_name}")

        if not Path(clean_path).exists():
            print(f"Image not found: {clean_path}")
            print("Tip: Use forward slashes or raw strings for paths")
            return []

        # Read the original image
        original_image = cv2.imread(clean_path)
        if original_image is None:
            print(f"Could not read image: {clean_path}")
            return []

        detections, result_obj = self.detect_defects(clean_path)

        print(f"Found {len(detections)} road defects:")
        for i, detection in enumerate(detections, 1):
            print(f"Defect {i}:")
            print(f"  Type: {detection['class']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  Severity: {detection['severity']}")
            print(f"  Bounding Box: {[round(coord, 2) for coord in detection['bbox']]}")

        if save_result and len(detections) > 0:
            output_path = f"results/detected_{image_name}"
            os.makedirs("results", exist_ok=True)

            # Create annotated image with circles
            annotated_image = self._draw_circle_annotations(original_image, detections)
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved: {output_path}")

            # Display the image with detections
            self.display_image_with_detections(annotated_image, detections)

        # Save detection log
        self._save_detection_log(clean_path, detections)

        return detections

    def display_image_with_detections(self, image, detections):
        """Display the image with detection results"""
        # Resize for better display if needed
        height, width = image.shape[:2]
        if width > 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Create a window and display the image
        cv2.imshow('Detection Results - Circles', image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _save_detection_log(self, image_path, detections):
        """Save detection results to JSON log"""
        log_entry = {
            'image': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'total_defects': len(detections)
        }

        log_file = "road_defect_test_log.json"
        try:
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)

            print(f"Log saved: {log_file}")

        except Exception as e:
            print(f"Could not save log: {e}")

    def test_on_test_set(self):
        """Test model on the entire test dataset"""
        test_images_path = "Dataset/test/images"

        if not os.path.exists(test_images_path):
            print(f"Test images directory not found: {test_images_path}")
            return

        image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Testing on {len(image_files)} test images...")

        total_detections = 0
        results = []
        detection_details = []

        for img_file in image_files:
            img_path = os.path.join(test_images_path, img_file)
            detections = self.process_image(img_path, save_result=True)
            total_detections += len(detections)

            # Store detailed results
            image_results = {
                'image': img_file,
                'defects_count': len(detections),
                'defects': [d['class'] for d in detections],
                'details': detections
            }
            results.append(image_results)
            detection_details.extend(detections)

        print(f"Test Summary:")
        print(f"  Total images tested: {len(image_files)}")
        print(f"  Total defects detected: {total_detections}")
        print(f"  Average defects per image: {total_detections / len(image_files):.2f}")

        # Print detailed results
        print("\nDetailed Results:")
        for result in results:
            if result['defects_count'] > 0:
                print(f"  {result['image']}: {result['defects_count']} defects - {', '.join(result['defects'])}")
            else:
                print(f"  {result['image']}: No defects detected")

        # Print defect statistics by type
        print("\nDefect Statistics by Type:")
        defect_counts = {cls: 0 for cls in self.class_names}
        for detection in detection_details:
            defect_counts[detection['class']] += 1

        for defect_type, count in defect_counts.items():
            print(f"  {defect_type}: {count}")

    def real_time_test(self, camera_index=0):
        """Real-time testing from webcam"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Could not open camera!")
            return

        print("Starting real-time testing...")
        print("  Press 'q' to quit")
        print("  Press 's' to save snapshot")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Run detection
            results = self.model(frame, conf=self.conf_threshold)

            # Process detections and draw circles
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()

                detections.append({
                    'class': self.class_names[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'severity': self._get_severity(confidence)
                })

            # Draw circles on the frame
            annotated_frame = self._draw_circle_annotations(frame, detections)

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Add overlay information
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Defects: {len(detections)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('Road Defect Tester - Circles', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/snapshot_{timestamp}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("Testing stopped")


# Usage menu:
if __name__ == "__main__":
    print("=" * 60)
    print("ROAD DEFECT TESTING SYSTEM WITH CIRCLE ANNOTATIONS")
    print("=" * 60)

    tester = RoadDefectTester(confidence_threshold=0.2)

    while True:
        print("\nTest Options:")
        print("1. Test single image")
        print("2. Test on entire test dataset")
        print("3. Real-time webcam test")
        print("4. Change confidence threshold")
        print("5. Exit")

        choice = input("Select option (1-5): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            # Use raw string to avoid path issues
            if image_path:
                tester.process_image(r"{}".format(image_path))

        elif choice == '2':
            print("Testing on full test dataset...")
            tester.test_on_test_set()

        elif choice == '3':
            tester.real_time_test()

        elif choice == '4':
            try:
                new_conf = float(input("Enter new confidence (0.1-0.9): "))
                if 0.1 <= new_conf <= 0.9:
                    tester.conf_threshold = new_conf
                    print(f"Confidence changed to: {new_conf}")
                else:
                    print("Invalid confidence value")
            except ValueError:
                print("Please enter a valid number")

        elif choice == '5':
            print("Exiting Road Defect Testing System")
            break

        else:
            print("Invalid option")