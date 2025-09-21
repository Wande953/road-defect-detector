# create_test_images.py
import cv2
import numpy as np


def create_alligator_test_image():
    """Create realistic alligator crack test image"""
    print("🛠️ Creating realistic alligator crack test image...")

    # Create asphalt background
    img = np.ones((640, 640, 3), dtype=np.uint8) * 80

    # Add realistic alligator cracking (network pattern)
    # Main cracks
    for i in range(200, 450, 20):
        cv2.line(img, (150, i), (500, i), (25, 25, 25), 2)
        cv2.line(img, (150 + i - 200, 200), (150 + i - 200, 450), (25, 25, 25), 2)

    # Add texture and noise
    noise = np.random.randint(-15, 15, (640, 640, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite("alligator_test.jpg", img)
    print("✅ Alligator test image created: alligator_test.jpg")


def create_pothole_test_image():
    """Create realistic pothole test image"""
    print("🛠️ Creating realistic pothole test image...")

    img = np.ones((640, 640, 3), dtype=np.uint8) * 90

    # Create pothole (dark circular area)
    center = (320, 320)
    cv2.circle(img, center, 40, (30, 30, 30), -1)

    # Add cracks around pothole
    for angle in range(0, 360, 45):
        x = int(center[0] + 50 * np.cos(np.radians(angle)))
        y = int(center[1] + 50 * np.sin(np.radians(angle)))
        cv2.line(img, center, (x, y), (20, 20, 20), 2)

    # Add texture
    noise = np.random.randint(-10, 10, (640, 640, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite("pothole_test.jpg", img)
    print("✅ Pothole test image created: pothole_test.jpg")


if __name__ == "__main__":
    create_alligator_test_image()
    create_pothole_test_image()