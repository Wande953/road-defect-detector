# detection_processor.py
from datetime import datetime
import threading
import random
import cv2
import numpy as np
from config import Config


class DetectionProcessor:
    def __init__(self):
        self.model = None  # No actual model
        self.data_lock = threading.Lock()
        self.detection_history = []
        self.current_stats = {
            'total_detections': 0,
            'by_class': {},
            'last_update': datetime.now().isoformat()
        }
        print("Running in test mode (no YOLO model - using simulated detections)")

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

    def process_detection(self, frame=None, image_path=None, location=None):
        """Simulate detection results for testing"""
        # Generate random detections for testing
        detections = []

        # Randomly decide if we should generate a detection (40% chance)
        if random.random() > 0.6:
            class_name = random.choice(Config.CLASS_NAMES)
            confidence = round(random.uniform(0.3, 0.9), 3)
            severity = self.get_severity(class_name, confidence)

            # Generate realistic bounding box coordinates
            if frame is not None:
                height, width = frame.shape[:2]
                x1 = random.randint(0, width - 100)
                y1 = random.randint(0, height - 100)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
            else:
                x1, y1, x2, y2 = [round(random.uniform(0, 100), 2) for _ in range(4)]

            detection = {
                'class': class_name,
                'class_id': Config.CLASS_NAMES.index(class_name),
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'severity': severity
            }

            if location:
                detection.update(location)

            detections.append(detection)

        with self.data_lock:
            for detection in detections:
                self.current_stats['total_detections'] += 1
                if detection['class'] in self.current_stats['by_class']:
                    self.current_stats['by_class'][detection['class']] += 1
                else:
                    self.current_stats['by_class'][detection['class']] = 1

            self.current_stats['last_update'] = datetime.now().isoformat()

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

    def draw_detections(self, frame, detections):
        """Draw detection boxes on the frame"""
        if frame is None:
            return frame

        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]

            # Choose color based on severity
            if detection['severity'] == 'High':
                color = (0, 0, 255)  # Red
            elif detection['severity'] == 'Medium':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame