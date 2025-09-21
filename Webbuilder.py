# config.py
import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
    UPLOAD_FOLDER = 'uploads'
    REPORTS_FOLDER = 'reports'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

    # Class names based on your model
    CLASS_NAMES = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']

    # Severity levels
    SEVERITY_LEVELS = {
        'Pothole': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Alligator': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Longitudinal': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Transverse': {'low': 0.3, 'medium': 0.6, 'high': 0.8}
    }

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.3


# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
os.makedirs('static/maps', exist_ok=True)

# camera_manager.py
import cv2
import threading
import time
import numpy as np


class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_lock = threading.Lock()

    def get_camera(self):
        """Get camera instance with thread safety"""
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                # Try different camera indices if 0 doesn't work
                for i in range(0, 4):
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        print(f"Camera found at index {i}")
                        # Set camera properties for better performance
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        break

                if self.camera.isOpened():
                    print("Camera initialized successfully")
                else:
                    print("Warning: No camera found")
                    # Create a dummy camera for testing
                    self.camera = cv2.VideoCapture(0)
                # Allow camera to warm up
                time.sleep(2)
            return self.camera

    def generate_placeholder_frame(self):
        """Generate a placeholder frame when no camera is available"""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No camera available", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder

    def cleanup(self):
        """Clean up camera resources"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                print("Camera resources cleaned up")


# detection_processor.py
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import threading
from settings.config import Config



class DetectionProcessor:
    def __init__(self):
        self.model = self.load_model()
        self.data_lock = threading.Lock()
        self.detection_history = []
        self.current_stats = {
            'total_detections': 0,
            'by_class': {},
            'last_update': datetime.now().isoformat()
        }

    def load_model(self):
        """Load the YOLO model"""
        try:
            model = YOLO('road_defect_final_model.pt')
            print("Custom model loaded successfully")
            return model
        except Exception as e:
            try:
                model = YOLO('yolov8n.pt')
                print("Using yolov8n as fallback")
                return model
            except Exception as e2:
                print(f"Error loading models: {e}, {e2}")
                return None

    def get_severity(self, class_name, confidence):
        """Determine severity level based on confidence"""
        if class_name in Config.SEVERITY_LEVELS:
            if confidence >= Config.SEVERITY_LEVELS[class_name]['high']:
                return 'High'
            elif confidence >= Config.SEVERITY_LEVELS[class_name]['medium']:
                return 'Medium'
            else:
                return 'Low'
        return 'Unknown'

    def process_detection(self, results, image_path=None, location=None):
        """Process detection results and update statistics"""
        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()

                class_name = Config.CLASS_NAMES[class_id] if class_id < len(Config.CLASS_NAMES) else f'Class_{class_id}'
                severity = self.get_severity(class_name, confidence)

                detection = {
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': round(confidence, 3),
                    'bbox': [round(coord, 2) for coord in bbox],
                    'timestamp': datetime.now().isoformat(),
                    'image_path': image_path,
                    'severity': severity
                }

                # Add location if available
                if location:
                    detection.update(location)

                detections.append(detection)

        # Update statistics with thread safety
        with self.data_lock:
            for detection in detections:
                self.current_stats['total_detections'] += 1
                if detection['class'] in self.current_stats['by_class']:
                    self.current_stats['by_class'][detection['class']] += 1
                else:
                    self.current_stats['by_class'][detection['class']] = 1

            self.current_stats['last_update'] = datetime.now().isoformat()

            # Add to history
            if detections:
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections,
                    'total': len(detections)
                })

        return detections, self.current_stats

    def get_stats(self):
        """Get current statistics"""
        with self.data_lock:
            return self.current_stats.copy()

    def get_history(self, limit=50):
        """Get detection history"""
        with self.data_lock:
            return self.detection_history[-limit:].copy()

    def get_filtered_defects(self, defect_type=None, severity=None, start_date=None, end_date=None):
        """Get defects with optional filters"""
        with self.data_lock:
            filtered_defects = []
            for detection in self.detection_history:
                for defect in detection['detections']:
                    # Apply filters
                    if defect_type and defect['class'] != defect_type:
                        continue
                    if severity and defect.get('severity') != severity:
                        continue
                    if start_date and defect['timestamp'] < start_date:
                        continue
                    if end_date and defect['timestamp'] > end_date:
                        continue

                    filtered_defects.append(defect)

            return filtered_defects