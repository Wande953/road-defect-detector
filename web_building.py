# web_dashboard.py
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import threading
import time
from datetime import datetime
import json
import os

import os
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import threading
import time
from datetime import datetime
import json

# Debug: Check current directory and template path
print("=== DEBUG INFORMATION ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Create template directory path
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
print(f"Template directory: {template_dir}")
print(f"Template directory exists: {os.path.exists(template_dir)}")
print(f"Index.html exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")

if os.path.exists(os.path.join(template_dir, 'index.html')):
    file_size = os.path.getsize(os.path.join(template_dir, 'index.html'))
    print(f"Index.html file size: {file_size} bytes")
else:
    print("Index.html does not exist!")

print("========================")

# Initialize Flask app with template folder
app = Flask(__name__, template_folder=template_dir)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Load your trained model (replace with your actual model path)
try:
    model = YOLO('road_defect_final_model.pt')
    print("Model loaded successfully")
except Exception as e:
    # Fallback to a pretrained model if custom model is not available
    print(f"Custom model not found: {e}, using yolov8n as fallback")
    model = YOLO('yolov8n.pt')

confidence_threshold = 0.3

# Thread-safe storage for detection history and stats
detection_history = []
current_stats = {
    'total_detections': 0,
    'by_class': {},
    'last_update': datetime.now().isoformat()
}
data_lock = threading.Lock()

# Update class names based on your model
class_names = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']

# Global camera variable with lazy initialization
camera = None
camera_lock = threading.Lock()


def get_camera():
    """Get camera instance with thread safety"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                # Try different camera indices if 0 doesn't work
                for i in range(1, 4):
                    camera = cv2.VideoCapture(i)
                    if camera.isOpened():
                        print(f"Camera found at index {i}")
                        break
            if camera.isOpened():
                print("Camera initialized successfully")
            else:
                print("Warning: No camera found")
            # Allow camera to warm up
            time.sleep(2)
        return camera


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', class_names=class_names)


@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    with data_lock:
        return jsonify(current_stats)


@app.route('/api/history')
def get_history():
    """Get detection history"""
    with data_lock:
        return jsonify(detection_history[-50:])  # Last 50 detections


def generate_frames():
    """Generate video frames with detections"""
    camera = get_camera()

    if not camera.isOpened():
        print("Camera not available for frame generation")
        return

    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Run detection
            results = model(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot()

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in frame generation: {e}")
            time.sleep(0.1)  # Prevent busy waiting on error


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def process_detection(results, image_path=None):
    """Process detection results and update statistics"""
    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].tolist()

            detection = {
                'class': class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}',
                'class_id': class_id,
                'confidence': round(confidence, 3),
                'bbox': [round(coord, 2) for coord in bbox],
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }

            detections.append(detection)

    # Update statistics with thread safety
    with data_lock:
        for detection in detections:
            current_stats['total_detections'] += 1
            if detection['class'] in current_stats['by_class']:
                current_stats['by_class'][detection['class']] += 1
            else:
                current_stats['by_class'][detection['class']] = 1

        current_stats['last_update'] = datetime.now().isoformat()

        # Add to history
        if detections:
            detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'total': len(detections)
            })

    # Send update via WebSocket
    if detections:
        socketio.emit('new_detection', {
            'detections': detections,
            'stats': current_stats
        })

    return detections


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    with data_lock:
        socketio.emit('initial_data', {
            'stats': current_stats,
            'history': detection_history[-20:]
        })


@socketio.on('process_image')
def handle_process_image(data):
    """Process uploaded image"""
    try:
        # Decode base64 image
        if 'image' not in data:
            socketio.emit('error', {'message': 'No image data provided'})
            return

        # Extract base64 data (handle both with and without data URL prefix)
        if data['image'].startswith('data:image'):
            image_data = base64.b64decode(data['image'].split(',')[1])
        else:
            image_data = base64.b64decode(data['image'])

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            socketio.emit('error', {'message': 'Failed to decode image'})
            return

        # Run detection
        results = model(img, conf=confidence_threshold)
        detections = process_detection(results, 'uploaded_image')

        # Send results back
        socketio.emit('image_results', {
            'detections': detections,
            'image_size': img.shape[:2]
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        socketio.emit('error', {'message': str(e)})


@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Process video file"""
    # This would handle video file uploads
    return jsonify({'status': 'video_processing_started'})


@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """Update model settings"""
    global confidence_threshold
    data = request.get_json()
    if data and 'confidence' in data:
        confidence_threshold = max(0.1, min(1.0, float(data['confidence'])))
    return jsonify({'confidence_threshold': confidence_threshold})


def cleanup_camera():
    """Clean up camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera resources cleaned up")


@app.teardown_appcontext
def shutdown_handler(exception=None):
    """Cleanup when application shuts down"""
    cleanup_camera()


if __name__ == '__main__':
    print("Starting Road Defect Detection Web Dashboard...")
    print("Open http://localhost:5000 in your browser")

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        cleanup_camera()