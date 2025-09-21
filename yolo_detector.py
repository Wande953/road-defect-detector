# yolo_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime


class RoadDefectDetector:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            self.class_names = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']
            self.performance_stats = {
                'overall_mAP50': 0.879,
                'class_accuracies': {
                    'Alligator': 0.556,
                    'Longitudinal': 0.983,
                    'Pothole': 0.994,
                    'Transverse': 0.984
                }
            }
            print(f"✅ YOLOv8 model loaded from: {model_path}")
            print(f"📊 Model Performance: 87.9% mAP50")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None

    def analyze_image(self, image_path, confidence_threshold=0.3):
        """Analyze image for road defects with performance insights"""
        if not self.model:
            return {"error": "Model not loaded"}

        if not os.path.exists(image_path):
            return {"error": f"Image file '{image_path}' does not exist"}

        try:
            # Class-specific confidence thresholds
            class_thresholds = {
                'Pothole': 0.25,  # Very reliable - lower threshold
                'Longitudinal': 0.3,  # Very reliable
                'Transverse': 0.4,  # Very reliable but fewer samples
                'Alligator': 0.5  # Less reliable - higher threshold
            }

            # Run inference
            results = self.model(image_path, conf=confidence_threshold, verbose=False)

            # Process results
            detections = []
            detection_count = {class_name: 0 for class_name in self.class_names}

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()

                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(
                            self.class_names) else f"Class_{class_id}"

                        # Apply class-specific threshold
                        threshold = class_thresholds.get(class_name, confidence_threshold)
                        if confidence >= threshold:
                            detection = {
                                "class": class_name,
                                "confidence": round(confidence, 2),
                                "bbox": [round(coord, 2) for coord in bbox],
                                "severity": self._calculate_severity(confidence, class_name),
                                "reliability": self._get_reliability(class_name)
                            }
                            detections.append(detection)

                            # Update count
                            if class_name in detection_count:
                                detection_count[class_name] += 1

            return {
                "success": True,
                "detections": detections,
                "total_detections": len(detections),
                "detection_count": detection_count,
                "performance_insights": self.performance_stats,
                "timestamp": datetime.now().isoformat(),
                "model_version": "road_defect_v1 (87.9% mAP50)"
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_severity(self, confidence, class_name):
        """Calculate severity based on confidence and class reliability"""
        if class_name == 'Pothole':
            if confidence > 0.9:
                return "CRITICAL"
            elif confidence > 0.7:
                return "High"
            else:
                return "Medium"

        elif class_name in ['Longitudinal', 'Transverse']:
            if confidence > 0.8:
                return "High"
            elif confidence > 0.5:
                return "Medium"
            else:
                return "Low"

        else:  # Alligator
            if confidence > 0.7:
                return "High (Verify)"
            elif confidence > 0.5:
                return "Medium (Verify)"
            else:
                return "Low (Verify)"

    def _get_reliability(self, class_name):
        """Get reliability rating for each class"""
        reliabilities = {
            'Pothole': 'Very High (99.4%)',
            'Longitudinal': 'Very High (98.3%)',
            'Transverse': 'Very High (98.4%)',
            'Alligator': 'Moderate (55.6%)'
        }
        return reliabilities.get(class_name, 'Unknown')


def get_detector():
    """Get the best available detector"""
    # Use the newly trained model
    trained_model_path = 'runs/detect/road_defect_v1/weights/best.pt'

    if os.path.exists(trained_model_path):
        print("✅ Loading newly trained YOLOv8 model (87.9% mAP50)")
        return RoadDefectDetector(trained_model_path)
    else:
        # Fallback to previous model
        previous_model = 'runs/detect/train20/weights/best.pt'
        if os.path.exists(previous_model):
            print("⚠ Using previous trained model (87.2% mAP50)")
            return RoadDefectDetector(previous_model)
        else:
            print("⚠ Using general YOLOv8 model")
            return RoadDefectDetector('yolov8n.pt')