import os

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# HTML content
html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Defect Detection Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }
        .container { 
            display: flex; 
            gap: 20px; 
            flex-wrap: wrap;
        }
        .video-container, .stats-container { 
            flex: 1; 
            min-width: 300px;
        }
        .video-feed { 
            width: 100%; 
            max-width: 640px;
            border: 2px solid #333; 
            border-radius: 8px;
        }
        .stats-panel { 
            background: white; 
            padding: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1 { color: #333; }
        h2 { color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .detection-item {
            background: #e8f4f8;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid #2196F3;
        }
        .upload-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #image-upload {
            margin: 10px 0;
            padding: 10px;
            border: 2px dashed #ccc;
            border-radius: 4px;
            width: 100%;
        }
        .results-container {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Road Defect Detection Dashboard</h1>

    <div class="container">
        <div class="video-container">
            <h2>Live Detection Feed</h2>
            <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Live Video Feed">
        </div>

        <div class="stats-container">
            <h2>Detection Statistics</h2>
            <div class="stats-panel">
                <h3>Total Detections: <span id="total-detections">0</span></h3>
                <div id="class-stats">
                    <p>No detections yet</p>
                </div>
            </div>

            <div class="upload-section">
                <h2>Upload Image for Analysis</h2>
                <input type="file" id="image-upload" accept="image/*">
                <div class="results-container" id="upload-results">
                    <p>Upload an image to see detection results</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Connect to SocketIO
        const socket = io();

        // Handle initial data
        socket.on('initial_data', function(data) {
            updateStats(data.stats);
        });

        // Handle new detections
        socket.on('new_detection', function(data) {
            updateStats(data.stats);
        });

        // Update statistics display
        function updateStats(stats) {
            document.getElementById('total-detections').textContent = stats.total_detections;

            let classStatsHtml = '';
            if (Object.keys(stats.by_class).length > 0) {
                classStatsHtml = '<h3>Detections by Class:</h3>';
                for (const [className, count] of Object.entries(stats.by_class)) {
                    classStatsHtml += `<div class="detection-item">${className}: ${count}</div>`;
                }
            } else {
                classStatsHtml = '<p>No detections yet</p>';
            }
            document.getElementById('class-stats').innerHTML = classStatsHtml;
        }

        // Handle image upload
        document.getElementById('image-upload').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    document.getElementById('upload-results').innerHTML = '<p>Processing image...</p>';
                    socket.emit('process_image', { image: event.target.result });
                };
                reader.readAsDataURL(file);
            }
        });

        // Handle image results
        socket.on('image_results', function(data) {
            let resultsHtml = '<h3>Detection Results:</h3>';
            if (data.detections.length > 0) {
                data.detections.forEach(detection => {
                    resultsHtml += `
                        <div class="detection-item">
                            <strong>${detection.class}</strong><br>
                            Confidence: ${(detection.confidence * 100).toFixed(1)}%<br>
                            Position: [${detection.bbox.map(x => x.toFixed(1)).join(', ')}]
                        </div>`;
                });
            } else {
                resultsHtml += '<p>No defects detected in this image</p>';
            }
            document.getElementById('upload-results').innerHTML = resultsHtml;
        });

        // Handle errors
        socket.on('error', function(data) {
            document.getElementById('upload-results').innerHTML = 
                `<p style="color: red;">Error: ${data.message}</p>`;
        });
    </script>
</body>
</html>'''

# Write the HTML file
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML file created successfully!")