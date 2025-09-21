# test_road_defect_model.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm


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

        detections, result_obj = self.detect_defects(clean_path)

        print(f"Found {len(detections)} road defects:")
        for i, detection in enumerate(detections, 1):
            print(f"Defect {i}:")
            print(f"  Type: {detection['class']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  Severity: {detection['severity']}")
            print(f"  Bounding Box: {[round(coord, 2) for coord in detection['bbox']]}")

        if save_result and result_obj and len(detections) > 0:
            output_path = f"results/detected_{image_name}"
            os.makedirs("results", exist_ok=True)

            # Create a copy of the image with detections to display
            result_img = result_obj.plot()
            cv2.imwrite(output_path, result_img)
            print(f"Result saved: {output_path}")

            # Display the image with detections
            self.display_image_with_detections(result_img, detections)

        # Save detection log
        self._save_detection_log(clean_path, detections)

        return detections

    def process_video(self, video_path, output_path=None, process_every_n_frames=5, show_preview=True):
        """Process video file for road defects"""
        clean_path = self._clean_path(video_path)
        video_name = Path(clean_path).name

        if not Path(clean_path).exists():
            print(f"Video not found: {clean_path}")
            return None

        print(f"Processing video: {video_name}")
        print(f"Processing every {process_every_n_frames} frames")

        # Open video file
        cap = cv2.VideoCapture(clean_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.1f}s")

        # Setup output video writer
        if output_path is None:
            output_path = f"results/processed_{video_name}"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Variables for tracking
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        all_detections = []
        start_time = time.time()

        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing Video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            # Process only every nth frame to save time
            if frame_count % process_every_n_frames == 0:
                processed_frames += 1

                # Run detection on current frame
                results = self.model(frame, conf=self.conf_threshold)
                annotated_frame = results[0].plot()

                # Extract detections
                frame_detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    bbox = box.xyxy[0].tolist()

                    detection = {
                        'class': self.class_names[class_id],
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'severity': self._get_severity(confidence),
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps
                    }
                    frame_detections.append(detection)
                    total_detections += 1

                all_detections.extend(frame_detections)

                # Add info overlay
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Defects: {len(frame_detections)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Write processed frame to output video
                out.write(annotated_frame)

                # Show preview if enabled
                if show_preview:
                    # Resize for display if needed
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))

                    cv2.imshow('Video Processing Preview', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break

            else:
                # Write original frame without processing
                out.write(frame)

        # Cleanup
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate processing statistics
        processing_time = time.time() - start_time
        real_time_factor = processing_time / duration

        print(f"\nVideo Processing Complete!")
        print(f"Processed {processed_frames} frames ({processed_frames / total_frames * 100:.1f}% of total)")
        print(f"Total defects detected: {total_detections}")
        print(f"Processing time: {processing_time:.1f}s (Real-time factor: {real_time_factor:.2f}x)")
        print(f"Output saved: {output_path}")

        # Generate video analysis report
        self._generate_video_report(video_name, all_detections, {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': duration,
            'processing_time': processing_time,
            'output_path': output_path
        })

        return {
            'output_path': output_path,
            'total_detections': total_detections,
            'detections': all_detections,
            'stats': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'processing_time': processing_time,
                'real_time_factor': real_time_factor
            }
        }

    def _generate_video_report(self, video_name, detections, video_info):
        """Generate detailed report for video processing"""
        report = {
            'video_name': video_name,
            'processing_date': datetime.now().isoformat(),
            'video_info': video_info,
            'detection_summary': {
                'total_defects': len(detections),
                'defects_by_type': {},
                'defects_by_severity': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'defects_over_time': []
            },
            'model_info': {
                'confidence_threshold': self.conf_threshold,
                'classes': self.class_names
            }
        }

        # Count defects by type and severity
        for detection in detections:
            defect_type = detection['class']
            severity = detection['severity']

            # Count by type
            if defect_type not in report['detection_summary']['defects_by_type']:
                report['detection_summary']['defects_by_type'][defect_type] = 0
            report['detection_summary']['defects_by_type'][defect_type] += 1

            # Count by severity
            report['detection_summary']['defects_by_severity'][severity] += 1

        # Save report
        os.makedirs("results/reports", exist_ok=True)
        report_filename = f"results/reports/{Path(video_name).stem}_report.json"

        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Detailed report saved: {report_filename}")

        # Print summary
        print("\n=== VIDEO ANALYSIS SUMMARY ===")
        print(f"Total defects: {report['detection_summary']['total_defects']}")
        print("\nDefects by type:")
        for defect_type, count in report['detection_summary']['defects_by_type'].items():
            print(f"  {defect_type}: {count}")

        print("\nDefects by severity:")
        for severity, count in report['detection_summary']['defects_by_severity'].items():
            print(f"  {severity}: {count}")

    def batch_process_videos(self, videos_directory, output_directory="results/videos"):
        """Process all videos in a directory"""
        if not os.path.exists(videos_directory):
            print(f"Directory not found: {videos_directory}")
            return

        os.makedirs(output_directory, exist_ok=True)

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = [f for f in os.listdir(videos_directory)
                       if f.lower().endswith(video_extensions)]

        print(f"Found {len(video_files)} videos to process")

        results = []
        for video_file in video_files:
            video_path = os.path.join(videos_directory, video_file)
            output_path = os.path.join(output_directory, f"processed_{video_file}")

            print(f"\nProcessing: {video_file}")
            result = self.process_video(video_path, output_path, show_preview=False)
            if result:
                results.append(result)

        return results

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
        cv2.imshow('Detection Results', image)
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
            annotated_frame = results[0].plot()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Add overlay information
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Defects: {len(results[0].boxes)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('Road Defect Tester', annotated_frame)

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
    print("ROAD DEFECT TESTING SYSTEM")
    print("=" * 60)

    tester = RoadDefectTester(confidence_threshold=0.5)

    while True:
        print("\nTest Options:")
        print("1. Test single image")
        print("2. Test on entire test dataset")
        print("3. Real-time webcam test")
        print("4. Process single video")
        print("5. Batch process videos from directory")
        print("6. Change confidence threshold")
        print("7. Exit")

        choice = input("Select option (1-7): ").strip()

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
            video_path = input("Enter video path: ").strip()
            if video_path:
                # Optional parameters
                process_every = input("Process every N frames (default: 5): ").strip()
                process_every = int(process_every) if process_every.isdigit() else 5

                show_preview = input("Show preview? (y/n, default: y): ").strip().lower()
                show_preview = show_preview != 'n'

                tester.process_video(r"{}".format(video_path),
                                     process_every_n_frames=process_every,
                                     show_preview=show_preview)

        elif choice == '5':
            directory_path = input("Enter directory path containing videos: ").strip()
            if directory_path:
                tester.batch_process_videos(r"{}".format(directory_path))

        elif choice == '6':
            try:
                new_conf = float(input("Enter new confidence (0.1-0.9): "))
                if 0.1 <= new_conf <= 0.9:
                    tester.conf_threshold = new_conf
                    print(f"Confidence changed to: {new_conf}")
                else:
                    print("Invalid confidence value")
            except ValueError:
                print("Please enter a valid number")

        elif choice == '7':
            print("Exiting Road Defect Testing System")
            break

        else:
            print("Invalid option")