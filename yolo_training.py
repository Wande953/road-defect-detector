import ultralytics
import yolov5
from ultralytics import YOLO

print(f"Using Ultralytics v{ultralytics.__version__}")

model = YOLO("yolov5n.yaml")

model.train(data = "data.yaml", epochs = 100)
model.train(
    data="data.yaml",
    epochs=50,            # Set a reasonable maximum
    save_period=5,        # Save every 5 epochs
    workers=4,            # Number of data loader workers
    patience=10,          # Stop if no improvement after 10 epochs
    imgsz=640
)