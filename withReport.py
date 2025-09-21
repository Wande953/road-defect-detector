# test_road_defect_model.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import folium
from folium.plugins import HeatMap
import webbrowser


class RoadDefectTester:
    def __init__(self, confidence_threshold=0.5):
        print("Initializing Road Defect Tester...")
        # Load your NEW trained model
        self.model = YOLO('road_defect_final_model.pt')  # Your new model
        self.conf_threshold = confidence_threshold
        self.class_names = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']  # NEW classes
        print("Model loaded successfully!")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"Detecting {len(self.class_names)} classes: {self.class_names}")

    def _clean_path(self, path_string):
        """Remove quotes and extra spaces from path"""
        return path_string.strip().strip('"').strip("'")

    def detect_defects(self, image_path):
        """Detect road defects in an image"""
        try:
            results = self.model(image_path, conf=self.conf_threshold)
            detections = []

            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()

                detections.append({
                    'class': self.class_names[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'severity': self._get_severity(confidence),
                    'timestamp': datetime.now().isoformat()
                })

            return detections, results[0]
        except Exception as e:
            print(f"Detection error: {e}")
            return [], None

    def _get_severity(self, confidence):
        """Convert confidence to severity level"""
        if confidence > 0.7:
            return "HIGH"
        elif confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def process_image(self, image_path, save_result=True):
        """Process single image and return results"""
        # Clean the path
        clean_path = self._clean_path(image_path)
        image_name = Path(clean_path).name

        print(f"Processing: {image_name}")

        if not Path(clean_path).exists():
            print(f"Image not found: {clean_path}")
            print("Tip: Use forward slashes or raw strings for paths")
            return []

        detections, result_obj = self.detect_defects(clean_path)

        print(f"Found {len(detections)} road defects:")
        for i, detection in enumerate(detections, 1):
            print(f"Defect {i}:")
            print(f"  Type: {detection['class']}")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print(f"  Severity: {detection['severity']}")
            print(f"  Bounding Box: {[round(coord, 2) for coord in detection['bbox']]}")

        if save_result and result_obj and len(detections) > 0:
            output_path = f"results/detected_{image_name}"
            os.makedirs("results", exist_ok=True)

            # Create a copy of the image with detections to display
            result_img = result_obj.plot()
            cv2.imwrite(output_path, result_img)
            print(f"Result saved: {output_path}")

            # Display the image with detections
            self.display_image_with_detections(result_img, detections)

        # Save detection log
        self._save_detection_log(clean_path, detections)

        return detections

    def process_video(self, video_path, output_path=None, process_every_n_frames=5, show_preview=True):
        """Process video file for road defects"""
        clean_path = self._clean_path(video_path)
        video_name = Path(clean_path).name

        if not Path(clean_path).exists():
            print(f"Video not found: {clean_path}")
            return None

        print(f"Processing video: {video_name}")
        print(f"Processing every {process_every_n_frames} frames")

        # Open video file
        cap = cv2.VideoCapture(clean_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.1f}s")

        # Setup output video writer
        if output_path is None:
            output_path = f"results/processed_{video_name}"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Variables for tracking
        frame_count = 0
        processed_frames = 0
        total_detections = 0
        all_detections = []
        start_time = time.time()

        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing Video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            # Process only every nth frame to save time
            if frame_count % process_every_n_frames == 0:
                processed_frames += 1

                # Run detection on current frame
                results = self.model(frame, conf=self.conf_threshold)
                annotated_frame = results[0].plot()

                # Extract detections
                frame_detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    bbox = box.xyxy[0].tolist()

                    detection = {
                        'class': self.class_names[class_id],
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'severity': self._get_severity(confidence),
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps
                    }
                    frame_detections.append(detection)
                    total_detections += 1

                all_detections.extend(frame_detections)

                # Add info overlay
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Defects: {len(frame_detections)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Write processed frame to output video
                out.write(annotated_frame)

                # Show preview if enabled
                if show_preview:
                    # Resize for display if needed
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))

                    cv2.imshow('Video Processing Preview', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break

            else:
                # Write original frame without processing
                out.write(frame)

        # Cleanup
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate processing statistics
        processing_time = time.time() - start_time
        real_time_factor = processing_time / duration

        print(f"\nVideo Processing Complete!")
        print(f"Processed {processed_frames} frames ({processed_frames / total_frames * 100:.1f}% of total)")
        print(f"Total defects detected: {total_detections}")
        print(f"Processing time: {processing_time:.1f}s (Real-time factor: {real_time_factor:.2f}x)")
        print(f"Output saved: {output_path}")

        # Generate video analysis report
        self._generate_video_report(video_name, all_detections, {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': duration,
            'processing_time': processing_time,
            'output_path': output_path
        })

        return {
            'output_path': output_path,
            'total_detections': total_detections,
            'detections': all_detections,
            'stats': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'processing_time': processing_time,
                'real_time_factor': real_time_factor
            }
        }

    def _generate_video_report(self, video_name, detections, video_info):
        """Generate detailed report for video processing"""
        report = {
            'video_name': video_name,
            'processing_date': datetime.now().isoformat(),
            'video_info': video_info,
            'detection_summary': {
                'total_defects': len(detections),
                'defects_by_type': {},
                'defects_by_severity': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'defects_over_time': []
            },
            'model_info': {
                'confidence_threshold': self.conf_threshold,
                'classes': self.class_names
            }
        }

        # Count defects by type and severity
        for detection in detections:
            defect_type = detection['class']
            severity = detection['severity']

            # Count by type
            if defect_type not in report['detection_summary']['defects_by_type']:
                report['detection_summary']['defects_by_type'][defect_type] = 0
            report['detection_summary']['defects_by_type'][defect_type] += 1

            # Count by severity
            report['detection_summary']['defects_by_severity'][severity] += 1

        # Save report
        os.makedirs("results/reports", exist_ok=True)
        report_filename = f"results/reports/{Path(video_name).stem}_report.json"

        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Detailed report saved: {report_filename}")

        # Print summary
        print("\n=== VIDEO ANALYSIS SUMMARY ===")
        print(f"Total defects: {report['detection_summary']['total_defects']}")
        print("\nDefects by type:")
        for defect_type, count in report['detection_summary']['defects_by_type'].items():
            print(f"  {defect_type}: {count}")

        print("\nDefects by severity:")
        for severity, count in report['detection_summary']['defects_by_severity'].items():
            print(f"  {severity}: {count}")

    def batch_process_videos(self, videos_directory, output_directory="results/videos"):
        """Process all videos in a directory"""
        if not os.path.exists(videos_directory):
            print(f"Directory not found: {videos_directory}")
            return

        os.makedirs(output_directory, exist_ok=True)

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = [f for f in os.listdir(videos_directory)
                       if f.lower().endswith(video_extensions)]

        print(f"Found {len(video_files)} videos to process")

        results = []
        for video_file in video_files:
            video_path = os.path.join(videos_directory, video_file)
            output_path = os.path.join(output_directory, f"processed_{video_file}")

            print(f"\nProcessing: {video_file}")
            result = self.process_video(video_path, output_path, show_preview=False)
            if result:
                results.append(result)

        return results

    def real_time_test(self, camera_index=0):
        """Real-time testing from webcam"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Could not open camera!")
            return

        print("Starting real-time testing...")
        print("  Press 'q' to quit")
        print("  Press 's' to save snapshot")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Add overlay information
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Defects: {len(results[0].boxes)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('Road Defect Tester', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/snapshot_{timestamp}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("Testing stopped")

    def real_time_smartphone_test(self, ip_address="10.11.50.86", port="8080"):
        """Real-time detection using smartphone camera"""
        # Construct the RTSP URL from your phone's IP
        rtsp_url = f"rtsp://{ip_address}:{port}/h264_ulaw.sdp"

        print(f"Connecting to smartphone camera: {rtsp_url}")
        print("Make sure both devices are on the same WiFi network")

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("Failed to connect to smartphone camera!")
            print("Please check:")
            print("1. Both devices on same WiFi")
            print("2. IP Webcam app is running")
            print("3. IP address is correct")
            return

        print("Connected to smartphone camera!")
        print("Press 'q' to quit, 's' to save snapshot")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost connection to camera")
                break

            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Add overlay information
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Conf: {self.conf_threshold}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Defects: {len(results[0].boxes)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(annotated_frame, "Smartphone Camera", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('Road Defect Detection - Smartphone Camera', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/smartphone_snapshot_{timestamp}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("Smartphone camera test stopped")

    def real_time_video_test(self, video_path):
        """Simulate real-time processing using a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return

        # Get video properties for realistic timing
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33  # ms between frames

        print("Simulating real-time detection...")
        print("Press 'q' to quit, 's' to save snapshot, 'p' to pause")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # Add info overlay
            cv2.putText(annotated_frame, "SIMULATION: Video File Input", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Defects: {len(results[0].boxes)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Road Defect Simulation', annotated_frame)

            # Control playback speed
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Pause
                cv2.waitKey(0)
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/simulation_snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot: {filename}")

        cap.release()
        cv2.destroyAllWindows()

    def generate_comprehensive_report(self, detection_data, report_type="all"):
        """
        Generate comprehensive reports based on detection data
        report_type: "all", "summary", "detailed", "maintenance", "alert"
        """
        report_generators = {
            "summary": self._generate_summary_report,
            "detailed": self._generate_detailed_report,
            "maintenance": self._generate_maintenance_report,
            "alert": self._generate_alert_report,
            "all": self._generate_all_reports
        }

        if report_type not in report_generators:
            print(f"Invalid report type. Available: {list(report_generators.keys())}")
            return

        return report_generators[report_type](detection_data)

    def _generate_all_reports(self, detection_data):
        """Generate all report types"""
        reports = {}
        reports['summary'] = self._generate_summary_report(detection_data)
        reports['detailed'] = self._generate_detailed_report(detection_data)
        reports['maintenance'] = self._generate_maintenance_report(detection_data)

        # Only generate alert report if there are high priority defects
        high_priority_defects = [d for d in detection_data if d['severity'] == 'HIGH']
        if high_priority_defects:
            reports['alert'] = self._generate_alert_report(detection_data)
        else:
            print("No high-priority defects found for alert report")

        return reports

    def _generate_summary_report(self, detection_data):
        """Generate summary report with statistics and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            'timestamp': timestamp,
            'total_defects': len(detection_data),
            'defects_by_type': self._count_defects_by_type(detection_data),
            'defects_by_severity': self._count_defects_by_severity(detection_data),
            'confidence_stats': self._calculate_confidence_stats(detection_data),
            'time_analysis': self._analyze_temporal_patterns(detection_data)
        }

        # Generate visualizations
        self._create_defect_distribution_chart(report_data['defects_by_type'], timestamp)
        self._create_severity_chart(report_data['defects_by_severity'], timestamp)
        self._create_confidence_histogram(detection_data, timestamp)

        # Save report
        report_path = f"reports/summary/summary_report_{timestamp}.json"
        os.makedirs("reports/summary", exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Summary report generated: {report_path}")
        return report_data

    def _generate_detailed_report(self, detection_data):
        """Generate detailed defect-by-defect report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        detailed_report = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_defects': len(detection_data),
                'report_id': f"DETAILED_{timestamp}"
            },
            'defects': detection_data,
            'analysis': {
                'most_common_defect': self._get_most_common_defect(detection_data),
                'highest_severity': self._get_highest_severity_defect(detection_data),
                'average_confidence': self._calculate_average_confidence(detection_data)
            }
        }

        report_path = f"reports/detailed/detailed_report_{timestamp}.json"
        os.makedirs("reports/detailed", exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)

        print(f"Detailed report generated: {report_path}")
        return detailed_report

    def _generate_maintenance_report(self, detection_data):
        """Generate maintenance action report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        maintenance_actions = self._generate_maintenance_actions(detection_data)

        report = {
            'report_id': f"MAINT_{timestamp}",
            'generation_date': datetime.now().isoformat(),
            'priority_defects': self._get_priority_defects(detection_data),
            'maintenance_actions': maintenance_actions,
            'estimated_costs': self._estimate_maintenance_costs(maintenance_actions),
            'timeline_recommendations': self._generate_timeline_recommendations(detection_data)
        }

        # Generate PDF version
        pdf_path = self._generate_maintenance_pdf(report, timestamp)

        report_path = f"reports/maintenance/maintenance_report_{timestamp}.json"
        os.makedirs("reports/maintenance", exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Maintenance report generated: {report_path}")
        print(f"PDF version: {pdf_path}")
        return report

    def _generate_alert_report(self, detection_data):
        """Generate alert report for immediate attention"""
        high_priority_defects = [d for d in detection_data if d['severity'] == 'HIGH']

        if not high_priority_defects:
            print("No high-priority defects found for alert report")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        alert_report = {
            'alert_id': f"ALERT_{timestamp}",
            'timestamp': datetime.now().isoformat(),
            'urgency_level': "IMMEDIATE",
            'high_priority_defects': high_priority_defects,
            'recommended_actions': [
                "Immediate inspection required",
                "Consider temporary road closure if severe",
                "Notify maintenance team immediately"
            ],
            'contact_points': [
                "Maintenance Department: +1-555-MAINT",
                "Safety Officer: safety@example.com",
                "Emergency Line: +1-555-EMERG"
            ]
        }

        report_path = f"reports/alerts/alert_report_{timestamp}.json"
        os.makedirs("reports/alerts", exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(alert_report, f, indent=2)

        # Send email alert if configured
        self._send_email_alert(alert_report)

        print(f"🚨 ALERT REPORT GENERATED: {report_path}")
        print(f"Found {len(high_priority_defects)} high-priority defects requiring immediate attention!")

        return alert_report

    def _generate_maintenance_pdf(self, report_data, timestamp):
        """Generate PDF version of maintenance report"""
        pdf_path = f"reports/maintenance/maintenance_report_{timestamp}.pdf"

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title = Paragraph("ROAD MAINTENANCE REPORT", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Metadata
        meta_text = f"""
        <b>Report ID:</b> {report_data['report_id']}<br/>
        <b>Generation Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Total Priority Defects:</b> {len(report_data['priority_defects'])}
        """
        story.append(Paragraph(meta_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Priority Defects
        story.append(Paragraph("<b>Priority Defects:</b>", styles['Heading2']))
        defect_data = [['Type', 'Severity', 'Confidence', 'Location']]
        for defect in report_data['priority_defects'][:10]:  # Show top 10
            defect_data.append([
                defect['class'],
                defect['severity'],
                f"{defect['confidence']:.1%}",
                str(defect.get('bbox', 'N/A'))
            ])

        defect_table = Table(defect_data)
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(defect_table)
        story.append(Spacer(1, 20))

        # Maintenance Actions
        story.append(Paragraph("<b>Recommended Maintenance Actions:</b>", styles['Heading2']))
        for action in report_data['maintenance_actions']:
            story.append(Paragraph(f"• {action}", styles['Normal']))

        story.append(Spacer(1, 20))

        # Timeline
        story.append(Paragraph("<b>Recommended Timeline:</b>", styles['Heading2']))
        for timeline in report_data['timeline_recommendations']:
            story.append(Paragraph(f"• {timeline}", styles['Normal']))

        doc.build(story)
        return pdf_path

    def _send_email_alert(self, alert_report):
        """Send email alert for high-priority defects"""
        # This is a template - configure with your email settings
        try:
            msg = MIMEMultipart()
            msg['From'] = 'road-defect-alert@example.com'
            msg['To'] = 'maintenance-team@example.com'
            msg['Subject'] = f"🚨 ROAD DEFECT ALERT - {alert_report['alert_id']}"

            body = f"""
            URGENT: High Priority Road Defects Detected

            Defects Found: {len(alert_report['high_priority_defects'])}
            Time: {alert_report['timestamp']}

            Immediate inspection required!

            Recommended Actions:
            {chr(10).join(alert_report['recommended_actions'])}

            Contact:
            {chr(10).join(alert_report['contact_points'])}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Uncomment and configure to actually send emails
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login('your-email@gmail.com', 'your-password')
            # server.send_message(msg)
            # server.quit()

            print("Email alert prepared (configure SMTP settings to actually send)")

        except Exception as e:
            print(f"Email configuration needed: {e}")

    # Helper methods for report generation
    def _count_defects_by_type(self, detection_data):
        counts = {cls: 0 for cls in self.class_names}
        for detection in detection_data:
            counts[detection['class']] += 1
        return counts

    def _count_defects_by_severity(self, detection_data):
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for detection in detection_data:
            counts[detection['severity']] += 1
        return counts

    def _calculate_confidence_stats(self, detection_data):
        confidences = [d['confidence'] for d in detection_data]
        if not confidences:
            return {'mean': 0, 'max': 0, 'min': 0}
        return {
            'mean': np.mean(confidences),
            'max': np.max(confidences),
            'min': np.min(confidences),
            'std': np.std(confidences)
        }

    def _analyze_temporal_patterns(self, detection_data):
        # Placeholder for temporal analysis
        return {"analysis": "Temporal patterns analysis would be implemented here"}

    def _get_most_common_defect(self, detection_data):
        if not detection_data:
            return "No defects"
        counts = self._count_defects_by_type(detection_data)
        return max(counts.items(), key=lambda x: x[1])[0]

    def _get_highest_severity_defect(self, detection_data):
        if not detection_data:
            return "No defects"
        return max(detection_data, key=lambda x: (x['severity'], x['confidence']))

    def _calculate_average_confidence(self, detection_data):
        if not detection_data:
            return 0
        return np.mean([d['confidence'] for d in detection_data])

    def _create_defect_distribution_chart(self, defect_counts, timestamp):
        plt.figure(figsize=(10, 6))
        defects = list(defect_counts.keys())
        counts = list(defect_counts.values())

        bars = plt.bar(defects, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A826'])
        plt.title('Defect Distribution by Type', fontweight='bold')
        plt.xlabel('Defect Type', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        os.makedirs("reports/charts", exist_ok=True)
        plt.savefig(f"reports/charts/defect_distribution_{timestamp}.png")
        plt.close()

    def _create_severity_chart(self, severity_counts, timestamp):
        plt.figure(figsize=(8, 8))
        labels = list(severity_counts.keys())
        sizes = list(severity_counts.values())
        colors = ['#FF6B6B', '#F9A826', '#4ECDC4']  # Red, Yellow, Green

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Defect Severity Distribution', fontweight='bold')

        plt.savefig(f"reports/charts/severity_distribution_{timestamp}.png")
        plt.close()

    def _create_confidence_histogram(self, detection_data, timestamp):
        confidences = [d['confidence'] for d in detection_data]
        if not confidences:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='#6A0DAD', edgecolor='black')
        plt.title('Confidence Score Distribution', fontweight='bold')
        plt.xlabel('Confidence Score', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.axvline(x=np.mean(confidences), color='red', linestyle='--',
                    label=f'Mean: {np.mean(confidences):.2f}')
        plt.legend()

        plt.savefig(f"reports/charts/confidence_histogram_{timestamp}.png")
        plt.close()

    def _generate_maintenance_actions(self, detection_data):
        actions = []
        for defect in detection_data:
            if defect['severity'] == 'HIGH':
                actions.append(
                    f"Immediate repair required for {defect['class']} (Confidence: {defect['confidence']:.1%})")
            elif defect['severity'] == 'MEDIUM':
                actions.append(f"Schedule repair for {defect['class']} within 7 days")
            else:
                actions.append(f"Monitor {defect['class']} - include in next maintenance cycle")
        return actions

    def _get_priority_defects(self, detection_data):
        return sorted(detection_data, key=lambda x: (x['severity'], x['confidence']), reverse=True)

    def _estimate_maintenance_costs(self, maintenance_actions):
        # Simple cost estimation - replace with actual cost data
        cost_per_action = {
            'Immediate repair': 500,
            'Schedule repair': 300,
            'Monitor': 50
        }

        total_cost = 0
        for action in maintenance_actions:
            if 'Immediate repair' in action:
                total_cost += cost_per_action['Immediate repair']
            elif 'Schedule repair' in action:
                total_cost += cost_per_action['Schedule repair']
            elif 'Monitor' in action:
                total_cost += cost_per_action['Monitor']

        return total_cost

    def _generate_timeline_recommendations(self, detection_data):
        timelines = []
        high_count = len([d for d in detection_data if d['severity'] == 'HIGH'])
        medium_count = len([d for d in detection_data if d['severity'] == 'MEDIUM'])

        if high_count > 0:
            timelines.append("IMMEDIATE: Address high-severity defects within 24 hours")
        if medium_count > 0:
            timelines.append("SHORT-TERM: Address medium-severity defects within 7 days")
        if len(detection_data) > 0:
            timelines.append("ONGOING: Continuous monitoring and scheduled maintenance")

        return timelines

    def display_image_with_detections(self, image, detections):
        """Display the image with detection results"""
        # Resize for better display if needed
        height, width = image.shape[:2]
        if width > 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Create a window and display the image
        cv2.imshow('Detection Results', image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _save_detection_log(self, image_path, detections):
        """Save detection results to JSON log"""
        log_entry = {
            'image': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'total_defects': len(detections)
        }

        log_file = "road_defect_test_log.json"
        try:
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)

            print(f"Log saved: {log_file}")

        except Exception as e:
            print(f"Could not save log: {e}")

    def test_on_test_set(self):
        """Test model on the entire test dataset"""
        test_images_path = "Dataset/test/images"

        if not os.path.exists(test_images_path):
            print(f"Test images directory not found: {test_images_path}")
            return

        image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Testing on {len(image_files)} test images...")

        total_detections = 0
        results = []
        detection_details = []

        for img_file in image_files:
            img_path = os.path.join(test_images_path, img_file)
            detections = self.process_image(img_path, save_result=True)
            total_detections += len(detections)

            # Store detailed results
            image_results = {
                'image': img_file,
                'defects_count': len(detections),
                'defects': [d['class'] for d in detections],
                'details': detections
            }
            results.append(image_results)
            detection_details.extend(detections)

        print(f"Test Summary:")
        print(f"  Total images tested: {len(image_files)}")
        print(f"  Total defects detected: {total_detections}")
        print(f"  Average defects per image: {total_detections / len(image_files):.2f}")

        # Print detailed results
        print("\nDetailed Results:")
        for result in results:
            if result['defects_count'] > 0:
                print(f"  {result['image']}: {result['defects_count']} defects - {', '.join(result['defects'])}")
            else:
                print(f"  {result['image']}: No defects detected")

        # Print defect statistics by type
        print("\nDefect Statistics by Type:")
        defect_counts = {cls: 0 for cls in self.class_names}
        for detection in detection_details:
            defect_counts[detection['class']] += 1

        for defect_type, count in defect_counts.items():
            print(f"  {defect_type}: {count}")


# Usage menu:
if __name__ == "__main__":
    print("=" * 60)
    print("ROAD DEFECT TESTING SYSTEM")
    print("=" * 60)

    tester = RoadDefectTester(confidence_threshold=0.5)

    while True:
        print("\nTest Options:")
        print("1. Test single image")
        print("2. Test on entire test dataset")
        print("3. Real-time testing")
        print("4. Process single video")
        print("5. Batch process videos from directory")
        print("6. Generate reports")
        print("7. Change confidence threshold")
        print("8. Exit")

        choice = input("Select option (1-8): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if image_path:
                tester.process_image(r"{}".format(image_path))

        elif choice == '2':
            print("Testing on full test dataset...")
            tester.test_on_test_set()

        elif choice == '3':
            print("\nReal-time Options:")
            print("a. Smartphone camera (WiFi)")
            print("b. Webcam (if available)")
            print("c. Video file simulation")

            rt_choice = input("Select real-time mode (a-c): ").strip().lower()

            if rt_choice == 'a':
                ip = input("Enter smartphone IP (default: 10.11.50.86): ").strip() or "10.11.50.86"
                port = input("Enter port (default: 8080): ").strip() or "8080"
                print(f"Using IP: {ip}:{port}")
                tester.real_time_smartphone_test(ip, port)
            elif rt_choice == 'b':
                tester.real_time_test()
            elif rt_choice == 'c':
                video_path = input("Enter video path for simulation: ").strip()
                if video_path:
                    tester.real_time_video_test(r"{}".format(video_path))
            else:
                print("Invalid choice")

        elif choice == '4':
            video_path = input("Enter video path: ").strip()
            if video_path:
                process_every = input("Process every N frames (default: 5): ").strip()
                process_every = int(process_every) if process_every.isdigit() else 5

                show_preview = input("Show preview? (y/n, default: y): ").strip().lower()
                show_preview = show_preview != 'n'

                tester.process_video(r"{}".format(video_path),
                                     process_every_n_frames=process_every,
                                     show_preview=show_preview)

        elif choice == '5':
            directory_path = input("Enter directory path containing videos: ").strip()
            if directory_path:
                tester.batch_process_videos(r"{}".format(directory_path))

        elif choice == '6':
            print("\nReport Generation Options:")
            print("a. Generate summary report")
            print("b. Generate detailed report")
            print("c. Generate maintenance report")
            print("d. Generate alert report")
            print("e. Generate all reports")

            report_choice = input("Select report type (a-e): ").strip().lower()

            report_types = {
                'a': 'summary',
                'b': 'detailed',
                'c': 'maintenance',
                'd': 'alert',
                'e': 'all'
            }

            if report_choice in report_types:
                # You need to have some detection data first
                image_path = input("Enter image path to analyze for report: ").strip()
                if image_path:
                    detections = tester.process_image(image_path, save_result=False)
                    if detections:
                        tester.generate_comprehensive_report(detections, report_types[report_choice])
            else:
                print("Invalid report choice")

        elif choice == '7':
            try:
                new_conf = float(input("Enter new confidence (0.1-0.9): "))
                if 0.1 <= new_conf <= 0.9:
                    tester.conf_threshold = new_conf
                    print(f"Confidence changed to: {new_conf}")
                else:
                    print("Invalid confidence value")
            except ValueError:
                print("Please enter a valid number")

        elif choice == '8':
            print("Exiting Road Defect Testing System")
            break

        else:
            print("Invalid option")