# dashboard.py
import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import tempfile
import os
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import threading
import time
import socket

# --- Page Config ---
st.set_page_config(page_title="Road Defect Inspector", layout="wide")
st.title("🛣️ Road Damage Detection System")
st.markdown("---")

# --- Configuration ---
CUSTOM_MODEL_PATH = "best.pt"

# --- Email Configuration ---
AUTHORITY_EMAILS = {
    "Public Works Department": "publicworks@city.gov",
    "Transportation Authority": "transport@city.gov",
    "Emergency Services": "emergency@city.gov"
}

EMAIL_PROVIDERS = {
    "Gmail": {"server": "smtp.gmail.com", "port": 587},
    "Outlook/Hotmail": {"server": "smtp.office365.com", "port": 587},
    "Yahoo": {"server": "smtp.mail.yahoo.com", "port": 587},
    "AOL": {"server": "smtp.aol.com", "port": 587},
    "Custom SMTP": {"server": "", "port": 587}
}

# --- Global variables for real-time detection ---
if 'stop_realtime' not in st.session_state:
    st.session_state.stop_realtime = False
if 'realtime_detection_data' not in st.session_state:
    st.session_state.realtime_detection_data = []
if 'camera_connected' not in st.session_state:
    st.session_state.camera_connected = False
if 'current_camera_url' not in st.session_state:
    st.session_state.current_camera_url = ""


# --- Model Loading ---
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    try:
        model = YOLO(CUSTOM_MODEL_PATH)
        if hasattr(model, 'names') and model.names is not None:
            st.success(f"✅ Custom Model Loaded! Detecting: {list(model.names.values())}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model. Error: {e}")
        st.stop()


# --- Email Notification Function ---
def send_email_notification(sender_email, sender_password, smtp_server, smtp_port, subject, message, recipient,
                            attachment_data=None, attachment_name="report.txt"):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        if attachment_data is not None:
            part = MIMEApplication(attachment_data, Name=attachment_name)
            part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()

        try:
            server.login(sender_email, sender_password)
        except Exception as login_error:
            st.warning(f"⚠️ Login failed: {login_error}. Trying to send without authentication...")

        server.send_message(msg)
        server.quit()

        return True

    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        st.info("📧 **Email Content (for manual sending):**")
        st.text(f"To: {recipient}")
        st.text(f"Subject: {subject}")
        st.text(f"Message: {message[:200]}...")
        return False


# --- Video Processing Function ---
def process_video(video_path, model, confidence_threshold=0.5):
    """Process video frame by frame and return annotated video"""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_data = []
    sample_frame = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the frame
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        result = results[0]

        # Draw bounding boxes on the frame
        annotated_frame = result.plot()

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # Save a sample frame for display (around 25% through the video)
        if frame_count == total_frames // 4 and sample_frame is None:
            sample_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Collect detection data for reporting
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                coords = [int(coord) for coord in box.xyxy[0].tolist()]
                detection_data.append({
                    "Defect Type": label,
                    "Confidence": f"{conf:.2%}",
                    "Frame": frame_count,
                    "Location": f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
                })

        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress:.1%})")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    return output_path, detection_data, sample_frame, total_frames, fps


# --- Test Port Function ---
def test_ports(ip_address):
    """Test common camera ports"""
    common_ports = [8080, 8888, 4747, 8554, 1935, 554, 80]
    results = []

    for port in common_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip_address, port))
            if result == 0:
                results.append(f"✅ Port {port}: OPEN")
            else:
                results.append(f"❌ Port {port}: CLOSED")
            sock.close()
        except:
            results.append(f"❌ Port {port}: ERROR")

    return results


# --- Real-time Detection Thread ---
def real_time_detection_thread(cap, model, confidence_threshold):
    """Thread for continuous real-time detection"""
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    fps_placeholder = st.empty()

    frame_count = 0
    start_time = time.time()

    while not st.session_state.stop_realtime:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from camera")
            break

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Run YOLO inference
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        result = results[0]

        # Draw bounding boxes
        annotated_frame = result.plot()

        # Convert BGR to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame with FPS
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True,
                                caption=f"Live Feed - FPS: {fps:.1f}")

        # Collect detection data
        current_detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                coords = [int(coord) for coord in box.xyxy[0].tolist()]

                detection = {
                    "Defect Type": label,
                    "Confidence": f"{conf:.2%}",
                    "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "Location": f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
                }
                current_detections.append(detection)
                st.session_state.realtime_detection_data.append(detection)

        # Update statistics
        if current_detections:
            stats_text = f"**Live Detections:** {len(current_detections)} defects\n"
            stats_text += f"**FPS:** {fps:.1f}\n"
            for det in current_detections:
                stats_text += f"- {det['Defect Type']} ({det['Confidence']})\n"
            stats_placeholder.info(stats_text)
        else:
            stats_placeholder.info(f"**Live Status:** No defects\n**FPS:** {fps:.1f}")

        # Small delay to prevent overwhelming the system
        time.sleep(0.03)

    cap.release()
    st.session_state.camera_connected = False


# --- Report Generation Function ---
def generate_detection_report(detection_data, source_info=None):
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content = f"ROAD DEFECT MAINTENANCE REPORT\n"
    report_content += f"Generated on: {report_date}\n"
    report_content += "=" * 60 + "\n\n"

    if source_info:
        report_content += f"Source analyzed: {source_info}\n\n"

    if detection_data:
        df = pd.DataFrame(detection_data)

        report_content += "DETECTED ROAD DEFECTS SUMMARY:\n"
        report_content += "=" * 60 + "\n\n"

        # Create table
        report_content += f"{'Defect Type':<15} {'Confidence':<12} {'Frame/Timestamp':<20}\n"
        report_content += f"{'-' * 15} {'-' * 12} {'-' * 20}\n"

        for _, row in df.iterrows():
            frame_info = str(row.get('Frame', row.get('Timestamp', 'N/A')))
            report_content += f"{row['Defect Type']:<15} {row['Confidence']:<12} {frame_info:<20}\n"

        # Add summary statistics
        report_content += "\n" + "=" * 60 + "\n"
        report_content += "SUMMARY STATISTICS:\n"
        report_content += "-" * 30 + "\n"

        total_defects = len(detection_data)
        defect_summary = df['Defect Type'].value_counts()

        report_content += f"Total defects detected: {total_defects}\n\n"
        report_content += "Defects by type:\n"
        for defect, count in defect_summary.items():
            report_content += f"- {defect}: {count}\n"

        # Overall severity assessment
        if total_defects > 10:
            overall_severity = "CRITICAL - Immediate road closure required"
        elif total_defects > 5:
            overall_severity = "HIGH - Immediate attention required"
        elif total_defects > 2:
            overall_severity = "MEDIUM - Schedule inspection within 48 hours"
        else:
            overall_severity = "LOW - Monitor situation"

        report_content += f"\nOVERALL SEVERITY LEVEL: {overall_severity}\n"

    else:
        report_content += "\nNO DEFECTS DETECTED\n"
        report_content += "No maintenance action required at this time.\n"

    report_content += "\n" + "=" * 60 + "\n"
    report_content += "End of Report - Automated Road Inspection System"

    return report_content


# --- Initialize the model ---
model = load_model()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Processing", "📱 Real-time Detection"])

# --- Image Analysis Tab ---
with tab1:
    st.header("Image Analysis")

    uploaded_file = st.file_uploader("Upload a road image", type=['jpg', 'jpeg', 'png'], key="image_upload")

    if uploaded_file is not None:
        original_img = Image.open(uploaded_file)
        st.image(original_img, caption="Uploaded Image", use_column_width=True)

        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, key="image_conf")

        if st.button("Analyze Image", type="primary", key="analyze_image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                original_img.save(tmp_file.name, format='JPEG')
                tmp_path = tmp_file.name

            with st.spinner('Analyzing for defects...'):
                results = model.predict(source=tmp_path, conf=confidence, imgsz=640)
                result = results[0]
                annotated_img = result.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            os.remove(tmp_path)

            st.subheader("Detection Results")
            st.image(annotated_img_rgb, use_column_width=True, caption="Detected Defects")

            detection_list = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    conf = float(box.conf)
                    coords = [int(coord) for coord in box.xyxy[0].tolist()]
                    detection_list.append({
                        "Defect Type": label,
                        "Confidence": f"{conf:.2%}",
                        "Bounding Box": f"{coords}"
                    })

            st.session_state.detection_data = detection_list

            if detection_list:
                df = pd.DataFrame(detection_list)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.write("**Summary:**")
                for label in df['Defect Type'].unique():
                    count = len(df[df['Defect Type'] == label])
                    st.write(f"- **{label}**: {count} found")
            else:
                st.info("No defects detected. Try adjusting the confidence threshold.")
                st.session_state.detection_data = []

# --- Video Processing Tab ---
with tab2:
    st.header("Video Processing")

    uploaded_video = st.file_uploader("Upload a road video", type=['mp4', 'avi', 'mov'], key="video_upload")

    if uploaded_video is not None:
        # Display original video
        st.subheader("Original Video")
        st.video(uploaded_video)

        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, key="video_conf")

        if st.button("Process Video", type="primary", key="process_video"):
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            # Process the video
            output_path, detection_data, sample_frame, total_frames, fps = process_video(video_path, model, confidence)

            # Display Results
            st.success("✅ Video processing complete!")

            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Processed Video")
                st.video(output_path)

                # Video metadata
                st.info(f"**Video Details:**")
                st.text(f"Duration: {total_frames / fps:.1f} seconds")
                st.text(f"Frames: {total_frames}")
                st.text(f"FPS: {fps}")
                st.text(f"Defects detected: {len(detection_data)}")

            with col2:
                if sample_frame is not None:
                    st.subheader("Sample Processed Frame")
                    st.image(sample_frame, caption="Sample frame with detections", use_column_width=True)

            # Detection Statistics
            st.subheader("Detection Statistics")

            if detection_data:
                df = pd.DataFrame(detection_data)

                # Create columns for statistics
                stat_col1, stat_col2, stat_col3 = st.columns(3)

                with stat_col1:
                    st.metric("Total Defects", len(detection_data))

                with stat_col2:
                    defect_types = df['Defect Type'].nunique()
                    st.metric("Unique Defect Types", defect_types)

                with stat_col3:
                    avg_confidence = np.mean([float(x.strip('%')) / 100 for x in df['Confidence']])
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")

                # Defects by type chart
                st.write("**Defects by Type:**")
                defect_counts = df['Defect Type'].value_counts()
                st.bar_chart(defect_counts)

                # Detailed detection data
                st.write("**Detailed Detection Data:**")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Defects over time (frames)
                st.write("**Defects Distribution Across Frames:**")
                defects_per_frame = df.groupby('Frame').size().reset_index(name='Defect Count')
                st.line_chart(defects_per_frame.set_index('Frame'))

                st.session_state.detection_data = detection_data
                st.session_state.video_processed = True

            else:
                st.info("No defects detected in the video.")
                st.session_state.detection_data = []
                st.session_state.video_processed = True

            # Download processed video
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Download Processed Video",
                    data=file,
                    file_name="processed_road_defects.mp4",
                    mime="video/mp4"
                )

            # Clean up
            os.unlink(video_path)
            os.unlink(output_path)

    # Show previous results if available
    elif 'video_processed' in st.session_state and st.session_state.video_processed:
        st.info("📁 Previously processed video results are available below.")

        if st.session_state.detection_data:
            df = pd.DataFrame(st.session_state.detection_data)

            st.subheader("Previous Video Analysis Results")

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Defects", len(st.session_state.detection_data))
            with col2:
                st.metric("Unique Types", df['Defect Type'].nunique())
            with col3:
                avg_conf = np.mean([float(x.strip('%')) / 100 for x in df['Confidence']])
                st.metric("Avg Confidence", f"{avg_conf:.1%}")

            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No defects were detected in the previous video analysis.")

# --- Real-time Detection Tab ---
with tab3:
    st.header("📱 Real-time Phone Camera Detection")

    # IP Camera Configuration with multiple options
    with st.expander("📡 Camera Connection Setup", expanded=True):
        st.info("""
        **Quick Setup Guide:**
        1. **On your phone:** Install 'IP Webcam' app (Android) or similar
        2. **Connect both devices** to the same WiFi network
        3. **Start the camera server** on your phone
        4. **Enter your phone's IP address** below (found in the app)
        5. **Try different connection methods** if one fails
        """)

        # Connection Method Selection
        connection_method = st.radio(
            "Connection Method:",
            ["Auto Detect", "IP Webcam (Android)", "DroidCam", "Custom URL"],
            horizontal=True
        )

        col1, col2 = st.columns(2)

        with col1:
            ip_address = st.text_input("Phone IP Address", "192.168.1.100",
                                       help="Your phone's local IP address (check WiFi settings)")
            port = st.number_input("Port", 8080, 9090, 8080,
                                   help="Common ports: 8080, 8888, 4747")

        with col2:
            username = st.text_input("Username (if required)", "")
            password = st.text_input("Password (if required)", "", type="password")

            # Test common ports button
            if st.button("🔍 Test Common Ports", key="test_ports"):
                port_results = test_ports(ip_address)
                st.write("**Port Test Results:**")
                for result in port_results:
                    st.write(result)

    # Generate camera URLs based on selected method
    camera_urls = []

    if connection_method == "Auto Detect":
        camera_urls = [
            f"http://{ip_address}:{port}/video",
            f"http://{ip_address}:{port}/stream",
            f"http://{ip_address}:{port}/videofeed",
            f"rtsp://{ip_address}:{port}/live.sdp",
            f"http://{ip_address}:{port}/ipcam",
            f"http://{ip_address}:{port}/mjpegfeed"
        ]
    elif connection_method == "IP Webcam (Android)":
        camera_urls = [
            f"http://{ip_address}:{port}/video",
            f"http://{ip_address}:{port}/videofeed",
            f"http://{ip_address}:{port}/ipcam",
        ]
    elif connection_method == "DroidCam":
        camera_urls = [
            f"http://{ip_address}:{port}/video",
            f"http://{ip_address}:4747/video",
        ]
    else:  # Custom URL
        custom_url = st.text_input("Custom Camera URL",
                                   f"http://{ip_address}:{port}/video",
                                   help="Enter the full URL provided by your camera app")
        camera_urls = [custom_url]

    # Display generated URLs for debugging
    with st.expander("🔗 Generated Camera URLs"):
        st.write("Trying these connection URLs:")
        for url in camera_urls:
            st.code(url)

    # Real-time detection controls
    st.subheader("Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, key="realtime_conf")

    # Connection buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        connect_btn = st.button("📡 Connect to Camera", type="primary", key="connect_camera")

    with col2:
        if st.session_state.camera_connected:
            stop_btn = st.button("⏹️ Stop Detection", type="secondary", key="stop_realtime_btn")
        else:
            stop_btn = False

    with col3:
        if st.button("🔄 Refresh Connection", key="refresh_conn"):
            st.session_state.stop_realtime = True
            st.rerun()

    # Connection status
    if st.session_state.camera_connected:
        st.success("🟢 Camera connected - Live detection active")
        st.balloons()
    else:
        st.warning("🔴 Camera not connected")

    # Try to connect when button is pressed
    if connect_btn and not st.session_state.camera_connected:
        with st.spinner("🔌 Trying to connect to camera..."):
            connected = False
            successful_url = ""

            for camera_url in camera_urls:
                try:
                    # Try to connect
                    if username and password:
                        # Handle authentication
                        cap = cv2.VideoCapture(camera_url)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                connected = True
                                successful_url = camera_url
                                break
                            cap.release()
                    else:
                        # Try without authentication
                        cap = cv2.VideoCapture(camera_url)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                connected = True
                                successful_url = camera_url
                                break
                            cap.release()

                except Exception as e:
                    continue

            if connected:
                st.success(f"✅ Connected successfully to: {successful_url}")
                st.session_state.camera_connected = True
                st.session_state.stop_realtime = False
                st.session_state.current_camera_url = successful_url

                # Start detection thread
                detection_thread = threading.Thread(
                    target=real_time_detection_thread,
                    args=(cap, model, confidence)
                )
                detection_thread.daemon = True
                detection_thread.start()

            else:
                st.error("❌ All connection attempts failed!")
                st.error("**Troubleshooting Tips:**")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.error("""
                    **On Your Phone:**
                    - Install 'IP Webcam' app
                    - Start server in the app
                    - Note the IP address shown
                    - Ensure WiFi is connected
                    """)

                with col_t2:
                    st.error("""
                    **On Your Computer:**
                    - Check phone IP is correct
                    - Both on same WiFi network
                    - Try different ports (8080, 8888)
                    - Disable firewall temporarily
                    """)

                st.info("""
                **Alternative Solutions:**
                - Use USB tethering instead of WiFi
                - Try a different camera app
                - Use the 'Test Common Ports' feature
                - Check phone's hotspot settings
                """)

    # Stop detection
    if stop_btn and st.session_state.camera_connected:
        st.session_state.stop_realtime = True
        st.session_state.camera_connected = False
        st.success("Detection stopped successfully")

    # Show real-time results
    if st.session_state.realtime_detection_data:
        st.subheader("Detection Results")

        # Convert to DataFrame for display
        df = pd.DataFrame(st.session_state.realtime_detection_data)

        # Show latest detections
        latest_detections = df.tail(20)  # Show last 20 detections

        st.write("**Latest Detections:**")
        st.dataframe(latest_detections, use_container_width=True, hide_index=True)

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(st.session_state.realtime_detection_data))
        with col2:
            unique_types = df['Defect Type'].nunique()
            st.metric("Unique Defect Types", unique_types)
        with col3:
            if not df.empty:
                avg_conf = np.mean([float(x.strip('%')) / 100 for x in df['Confidence']])
                st.metric("Avg Confidence", f"{avg_conf:.1%}")

        # Defects over time chart
        st.write("**Detection Timeline:**")
        df['Time'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')
        defects_over_time = df.groupby(df['Time'].dt.floor('S')).size()
        st.line_chart(defects_over_time)

        # Clear results button
        if st.button("🧹 Clear Results", key="clear_results"):
            st.session_state.realtime_detection_data = []
            st.rerun()

# --- Reporting Section (Collapsible) ---
with st.expander("📊 Reporting & Alerts", expanded=False):
    st.header("Generate Reports & Send Alerts")

    # Email configuration
    with st.expander("✉️ Email Settings", expanded=False):
        email_provider = st.selectbox(
            "Your Email Provider:",
            list(EMAIL_PROVIDERS.keys())
        )

        if email_provider == "Custom SMTP":
            smtp_server = st.text_input("SMTP Server:", "smtp.yourprovider.com")
            smtp_port = st.number_input("SMTP Port:", 587, 587)
        else:
            smtp_server = EMAIL_PROVIDERS[email_provider]["server"]
            smtp_port = EMAIL_PROVIDERS[email_provider]["port"]

        sender_email = st.text_input("Your Email Address:", "your.email@provider.com")
        sender_password = st.text_input("Your Email Password:", type="password")

    selected_authority = st.selectbox(
        "Notify Authority:",
        list(AUTHORITY_EMAILS.keys())
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📋 Generate Report", key="gen_report"):
            detection_data = st.session_state.get('detection_data', []) or st.session_state.get(
                'realtime_detection_data', [])
            if detection_data:
                source_info = "Real-time IP Camera Analysis" if st.session_state.camera_connected else "Image/Video Analysis"
                report_text = generate_detection_report(detection_data, source_info)

                st.download_button(
                    label="⬇️ Download Report",
                    data=report_text,
                    file_name=f"road_maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_report"
                )

                st.text_area("Report Preview", report_text, height=200)
            else:
                st.warning("Please analyze an image or video first to generate a report.")

    with col2:
        if st.button("🚨 Send Alert", type="secondary", key="send_alert"):
            detection_data = st.session_state.get('detection_data', []) or st.session_state.get(
                'realtime_detection_data', [])
            if detection_data:
                subject = f"Road Defects Detected - {datetime.now().strftime('%Y-%m-%d')}"
                message = f"""
                Automated Road Inspection System Alert:

                Road defects have been detected requiring attention.

                Detection Summary:
                - Total defects found: {len(detection_data)}
                - Defect types: {', '.join(set([d['Defect Type'] for d in detection_data]))}

                Please review the detailed report for more information.

                This is an automated message from the Road Damage Detection System.
                """

                report_text = generate_detection_report(detection_data)
                report_bytes = report_text.encode('utf-8')

                if send_email_notification(sender_email, sender_password, smtp_server, smtp_port,
                                           subject, message, AUTHORITY_EMAILS[selected_authority],
                                           report_bytes, "road_defect_report.txt"):
                    st.success(f"✅ Alert sent to {selected_authority}")
            else:
                st.warning("Please analyze an image or video first before sending alerts.")

st.markdown("---")
st.caption("Road Defect Detection System | Powered by YOLOv8 and Streamlit")