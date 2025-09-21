# check_model_performance.py
from ultralytics import YOLO


def check_model_performance():
    model = YOLO('runs/detect/train20/weights/best.pt')

    # Test the model on validation set
    results = model.val(data='data.yaml', split='val')

    print("📊 Model Performance Summary:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")

    print("\n🎯 Class-wise Performance:")
    for i, class_name in model.names.items():
        print(f"{class_name}: mAP50={results.box.map50:.3f}")


if __name__ == "__main__":
    check_model_performance()