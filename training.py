from ultralytics import YOLO
from pathlib import Path

# Load your trained model
model = YOLO('runs/detect/train192/weights/best.pt')

# Test on an image
image_path = r"C:\Users\202234042\Documents\Dataset\test\images\img-86_jpg.rf.fb31876fd760109b02647f28c943583d.jpg"
results = model(image_path)  # REMOVE the duplicate 'results = model' part

# Display results
results[0].show()

# Save results
results[0].save('detection_result.jpg')

# Get detection information
print(f"Number of detections: {len(results[0].boxes)}")
if len(results[0].boxes) > 0:
    for i, box in enumerate(results[0].boxes):
        print(f"Detection {i+1}:")
        print(f"  Class: {box.cls.item()}")
        print(f"  Confidence: {box.conf.item():.3f}")
        print(f"  Coordinates: {box.xyxy[0].tolist()}")
else:
    print("No objects detected!")
    print("This confirms the training issue - the model didn't learn to detect anything.")