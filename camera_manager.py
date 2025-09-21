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