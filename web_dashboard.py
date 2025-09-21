# web_dashboard.py
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import threading
import time
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename

# Import your custom modules
from config import Config
from camera_manager import CameraManager
from detection_processor import DetectionProcessor
from report_generator import ReportGenerator
from data_manager import DataManager

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Initialize components
camera_manager = CameraManager()
detection_processor = DetectionProcessor()
report_generator = ReportGenerator()
data_manager = DataManager()

# Global variables
confidence_threshold = Config.CONFIDENCE_THRESHOLD


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', class_names=Config.CLASS_NAMES)


@app.route('/dashboard')
def dashboard():
    """Dashboard with visualizations"""
    stats = detection_processor.get_stats()
    return render_template('dashboard.html',
                           defect_counts=stats['by_class'],
                           total_detections=stats['total_detections'],
                           class_names=Config.CLASS_NAMES)


@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(detection_processor.get_stats())


@app.route('/api/history')
def get_history():
    """Get detection history"""
    return jsonify(detection_processor.get_history(50))


def generate_frames():
    """Generate video frames with simulated detections"""
    camera = camera_manager.get_camera()

    if not camera.isOpened():
        while True:
            placeholder = camera_manager.generate_placeholder_frame()
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        return

    while True:
        try:
            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue

            # Process simulated detections
            detections, stats = detection_processor.process_detection(frame)

            # Draw detections on the frame
            annotated_frame = detection_processor.draw_detections(frame.copy(), detections)

            # Send detection updates via WebSocket
            if detections:
                socketio.emit('new_detection', {
                    'detections': detections,
                    'stats': stats
                })

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    stats = detection_processor.get_stats()
    history = detection_processor.get_history(20)
    socketio.emit('initial_data', {
        'stats': stats,
        'history': history
    })


@socketio.on('process_image')
def handle_process_image(data):
    """Process uploaded image"""
    try:
        # Decode base64 image
        if 'image' not in data:
            socketio.emit('error', {'message': 'No image data provided'})
            return

        # Extract base64 data
        if data['image'].startswith('data:image'):
            image_data = base64.b64decode(data['image'].split(',')[1])
        else:
            image_data = base64.b64decode(data['image'])

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            socketio.emit('error', {'message': 'Failed to decode image'})
            return

        # Get location if provided
        location = data.get('location', {})

        # Process simulated detections
        detections, stats = detection_processor.process_detection(img, 'uploaded_image', location)

        # Send results back
        socketio.emit('image_results', {
            'detections': detections,
            'image_size': img.shape[:2]
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        socketio.emit('error', {'message': str(e)})


@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """Update model settings"""
    global confidence_threshold
    data = request.get_json()
    if data and 'confidence' in data:
        confidence_threshold = max(0.1, min(1.0, float(data['confidence'])))
    return jsonify({'confidence_threshold': confidence_threshold})


@app.teardown_appcontext
def shutdown_handler(exception=None):
    """Cleanup when application shuts down"""
    camera_manager.cleanup()


if __name__ == '__main__':
    print("Starting Road Defect Detection Web Dashboard...")
    print("Running in TEST MODE - using simulated detections")
    print("Open http://localhost:5000 in your browser")

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        camera_manager.cleanup()