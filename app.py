# app.py
from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
import json
import random
import csv
from pathlib import Path

app = Flask(__name__)

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/alerts', exist_ok=True)


class AdvancedDetector:
    def __init__(self):
        self.detection_history = []
        self.maintenance_reports = []

    def analyze_image(self, image_path, confidence_threshold=0.3, location=None):
        # Simulate analysis with realistic detections
        classes = ['Pothole', 'Longitudinal', 'Transverse', 'Alligator']
        detections = []

        num_detections = random.randint(1, 6)
        for _ in range(num_detections):
            class_name = random.choice(classes)
            confidence = round(random.uniform(0.4, 0.95), 2)

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "severity": self._calculate_severity(confidence, class_name),
                "reliability": self._get_reliability(class_name),
                "size": f"{random.randint(5, 50)}cm",
                "repair_priority": self._get_repair_priority(class_name, confidence)
            })

        detection_count = {cls: sum(1 for d in detections if d['class'] == cls) for cls in classes}

        result = {
            "success": True,
            "detections": detections,
            "total_detections": num_detections,
            "detection_count": detection_count,
            "performance_insights": {
                "overall_mAP50": 0.879,
                "class_accuracies": {
                    "Pothole": 0.994,
                    "Longitudinal": 0.983,
                    "Transverse": 0.984,
                    "Alligator": 0.556
                }
            },
            "timestamp": datetime.now().isoformat(),
            "model_version": "road_defect_v1 (87.9% mAP50)",
            "location": location or self._generate_random_location(),
            "image_metadata": {
                "resolution": "1920x1080",
                "file_size": f"{random.randint(500, 5000)}KB"
            }
        }

        # Save to history
        self.detection_history.append(result)
        self._generate_maintenance_report(result)
        self._check_for_alerts(result)

        return result

    def _calculate_severity(self, confidence, class_name):
        if class_name == 'Pothole':
            if confidence > 0.8:
                return "CRITICAL"
            elif confidence > 0.6:
                return "High"
            else:
                return "Medium"
        else:
            if confidence > 0.7:
                return "High"
            elif confidence > 0.5:
                return "Medium"
            else:
                return "Low"

    def _get_reliability(self, class_name):
        reliabilities = {
            'Pothole': 'Very High (99.4%)',
            'Longitudinal': 'Very High (98.3%)',
            'Transverse': 'Very High (98.4%)',
            'Alligator': 'Moderate (55.6%)'
        }
        return reliabilities.get(class_name, 'Unknown')

    def _get_repair_priority(self, class_name, confidence):
        if class_name == 'Pothole' and confidence > 0.7:
            return "URGENT (24h)"
        elif confidence > 0.8:
            return "HIGH (72h)"
        elif confidence > 0.6:
            return "MEDIUM (1 week)"
        else:
            return "LOW (2 weeks)"

    def _generate_random_location(self):
        # Simulate GPS coordinates around a city
        lat = 40.7128 + random.uniform(-0.1, 0.1)
        lng = -74.0060 + random.uniform(-0.1, 0.1)
        return {
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "address": f"{random.randint(1, 1000)} Main St, City Center"
        }

    def _generate_maintenance_report(self, result):
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": result['timestamp'],
            "location": result['location'],
            "total_defects": result['total_detections'],
            "critical_issues": sum(1 for d in result['detections'] if d['severity'] == 'CRITICAL'),
            "defect_breakdown": result['detection_count'],
            "estimated_repair_cost": self._estimate_repair_cost(result),
            "recommended_actions": self._generate_actions(result)
        }

        self.maintenance_reports.append(report)

        # Save to CSV
        self._save_report_to_csv(report)

        return report

    def _estimate_repair_cost(self, result):
        base_cost = 0
        for detection in result['detections']:
            if detection['class'] == 'Pothole':
                base_cost += random.randint(200, 1000)
            else:
                base_cost += random.randint(50, 300)
        return f"${base_cost}"

    def _generate_actions(self, result):
        actions = []
        for detection in result['detections']:
            if detection['class'] == 'Pothole':
                actions.append(f"Fill pothole ({detection['size']}) - Priority: {detection['repair_priority']}")
            elif detection['class'] == 'Longitudinal':
                actions.append(f"Seal longitudinal crack - Priority: {detection['repair_priority']}")
            elif detection['class'] == 'Transverse':
                actions.append(f"Repair transverse crack - Priority: {detection['repair_priority']}")
            else:
                actions.append(f"Address alligator cracking - Priority: {detection['repair_priority']}")
        return actions

    def _save_report_to_csv(self, report):
        csv_file = 'data/reports/maintenance_reports.csv'
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Report ID', 'Date', 'Location', 'Total Defects', 'Critical Issues', 'Estimated Cost'])

            writer.writerow([
                report['report_id'],
                report['timestamp'],
                f"{report['location']['latitude']}, {report['location']['longitude']}",
                report['total_defects'],
                report['critical_issues'],
                report['estimated_repair_cost']
            ])

    def _check_for_alerts(self, result):
        critical_detections = [d for d in result['detections'] if d['severity'] == 'CRITICAL']
        if critical_detections:
            alert = {
                "alert_id": f"ALT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "timestamp": result['timestamp'],
                "location": result['location'],
                "critical_issues": len(critical_detections),
                "message": f"🚨 {len(critical_detections)} CRITICAL road defects detected!",
                "defects": [f"{d['class']} ({d['size']})" for d in critical_detections],
                "status": "PENDING"
            }

            # Save alert
            alert_file = f"data/alerts/{alert['alert_id']}.json"
            with open(alert_file, 'w') as f:
                json.dump(alert, f, indent=2)

            print(f"🚨 ALERT GENERATED: {alert['message']}")


# Initialize detector
detector = AdvancedDetector()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Get location data if provided
        location_data = request.form.get('location')
        location = json.loads(location_data) if location_data else None

        # Save uploaded file
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        image_path = os.path.join('uploads', filename)
        file.save(image_path)

        # Analyze with detector
        results = detector.analyze_image(image_path, location=location)
        results['filename'] = filename
        results['original_filename'] = file.filename

        # Save results
        results_json_path = os.path.join('static', 'results', f'{filename}.json')
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/stats')
def get_statistics():
    stats = {
        'model_performance': {
            'overall_accuracy': 0.879,
            'precision': 0.854,
            'recall': 0.852,
            'training_time': '7.24 hours',
            'epochs': 100,
            'dataset_size': {
                'train': 1279,
                'val': 339,
                'test': 191
            }
        },
        'class_performance': {
            'Pothole': {'accuracy': 0.994, 'color': '#10B981'},
            'Longitudinal': {'accuracy': 0.983, 'color': '#3B82F6'},
            'Transverse': {'accuracy': 0.984, 'color': '#8B5CF6'},
            'Alligator': {'accuracy': 0.556, 'color': '#F59E0B'}
        },
        'system_stats': {
            'total_analyses': len(detector.detection_history),
            'total_defects_detected': sum(r['total_detections'] for r in detector.detection_history),
            'critical_alerts': len([f for f in os.listdir('data/alerts') if f.endswith('.json')]),
            'maintenance_reports': len(detector.maintenance_reports)
        }
    }
    return jsonify(stats)


@app.route('/reports')
def get_reports():
    reports = detector.maintenance_reports[-10:]  # Last 10 reports
    return jsonify(reports)


@app.route('/alerts')
def get_alerts():
    alerts = []
    alerts_dir = 'data/alerts'
    if os.path.exists(alerts_dir):
        alert_files = sorted(os.listdir(alerts_dir), reverse=True)[:5]
        for file in alert_files:
            if file.endswith('.json'):
                with open(os.path.join(alerts_dir, file), 'r') as f:
                    alerts.append(json.load(f))
    return jsonify(alerts)


@app.route('/map-data')
def get_map_data():
    map_data = []
    for result in detector.detection_history[-50:]:  # Last 50 detections
        if result.get('location'):
            map_data.append({
                'lat': result['location']['latitude'],
                'lng': result['location']['longitude'],
                'defects': result['total_detections'],
                'critical': sum(1 for d in result['detections'] if d['severity'] == 'CRITICAL'),
                'timestamp': result['timestamp']
            })
    return jsonify(map_data)


@app.route('/download-report/<report_id>')
def download_report(report_id):
    report = next((r for r in detector.maintenance_reports if r['report_id'] == report_id), None)
    if report:
        return jsonify(report)
    return jsonify({'error': 'Report not found'})


if __name__ == '__main__':
    print("🚀 Starting Advanced Road Defect Detection Dashboard")
    print("📊 Features: Analytics, Mapping, Reports, Alerts")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')