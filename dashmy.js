// static/js/dashboard.js

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Road Defect Detection Dashboard initialized');
    initCharts();
    setupEventListeners();
    connectWebSocket();
});

// Initialize charts with empty data
function initCharts() {
    // Defect distribution pie chart
    const defectCtx = document.getElementById('defectChart').getContext('2d');
    window.defectChart = new Chart(defectCtx, {
        type: 'pie',
        data: {
            labels: ['Alligator', 'Longitudinal', 'Pothole', 'Transverse'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
                title: {
                    display: true,
                    text: 'Defect Distribution'
                }
            }
        }
    });

    // Severity distribution bar chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    window.severityChart = new Chart(severityCtx, {
        type: 'bar',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                label: 'Number of Defects',
                data: [0, 0, 0],
                backgroundColor: ['#dc3545', '#ffc107', '#28a745']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Defects'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Defects by Severity Level'
                }
            }
        }
    });
}

// Set up event listeners for buttons and controls
function setupEventListeners() {
    // Confidence slider
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValue = document.getElementById('confidenceValue');

    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
    }

    // Image upload handling
    const imageUpload = document.getElementById('imageUpload');
    if (imageUpload) {
        imageUpload.addEventListener('change', handleImageUpload);
    }

    // Location checkbox
    const useLocation = document.getElementById('useLocation');
    const locationFields = document.getElementById('locationFields');

    if (useLocation && locationFields) {
        useLocation.addEventListener('change', function() {
            locationFields.style.display = this.checked ? 'block' : 'none';
        });
    }
}

// Handle image upload and processing
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const imageData = e.target.result;

        // Get location data if enabled
        let locationData = {};
        const useLocation = document.getElementById('useLocation');
        if (useLocation && useLocation.checked) {
            const lat = document.getElementById('latitude').value;
            const lon = document.getElementById('longitude').value;
            if (lat && lon) {
                locationData = { lat: parseFloat(lat), lon: parseFloat(lon) };
            }
        }

        // Send image to server for processing
        if (window.socket) {
            window.socket.emit('process_image', {
                image: imageData,
                location: locationData
            });
        }
    };
    reader.readAsDataURL(file);
}

// Connect to Flask-SocketIO
function connectWebSocket() {
    window.socket = io();

    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('new_detection', function(data) {
        console.log('New detection:', data);
        updateCharts(data.stats);
        updateDetectionTable(data.detections);
    });

    socket.on('initial_data', function(data) {
        console.log('Initial data received:', data);
        updateCharts(data.stats);

        // Flatten all detections from history
        const allDetections = data.history.flatMap(item => item.detections);
        updateDetectionTable(allDetections);
    });

    socket.on('image_results', function(data) {
        console.log('Image processing results:', data);
        showImageResults(data);
    });

    socket.on('error', function(data) {
        console.error('Error:', data);
        alert('Error: ' + data.message);
    });

    socket.on('alert', function(data) {
        console.log('Alert received:', data);
        showAlert(data.message);
    });
}

// Update charts with new data
function updateCharts(stats) {
    // Update defect distribution chart
    const defectLabels = Object.keys(stats.by_class);
    const defectData = defectLabels.map(label => stats.by_class[label]);

    window.defectChart.data.labels = defectLabels;
    window.defectChart.data.datasets[0].data = defectData;
    window.defectChart.update();

    // Update total detections counter
    document.getElementById('totalDetections').textContent = stats.total_detections;
}

// Update the detection table with new detections
function updateDetectionTable(detections) {
    const tableBody = document.getElementById('detectionsTable');

    // Clear existing rows if we have new detections
    if (detections && detections.length > 0) {
        tableBody.innerHTML = '';
    } else {
        tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No detections yet</td></tr>';
        return;
    }

    // Show latest 10 detections (most recent first)
    const recentDetections = detections.slice(-10).reverse();

    recentDetections.forEach(detection => {
        const row = document.createElement('tr');

        // Format timestamp
        const date = new Date(detection.timestamp);
        const timeString = date.toLocaleTimeString();

        // Add severity class for styling
        row.classList.add(`severity-${detection.severity.toLowerCase()}`);

        row.innerHTML = `
            <td>${timeString}</td>
            <td>${detection.class}</td>
            <td>${(detection.confidence * 100).toFixed(1)}%</td>
            <td><span class="badge bg-${getSeverityColor(detection.severity)}">${detection.severity}</span></td>
        `;

        tableBody.appendChild(row);
    });
}

// Show image processing results
function showImageResults(data) {
    // Create a results section
    let resultsDiv = document.getElementById('imageResults');
    if (!resultsDiv) {
        resultsDiv = document.createElement('div');
        resultsDiv.id = 'imageResults';
        resultsDiv.className = 'mt-3';
        document.querySelector('.card-body').appendChild(resultsDiv);
    }

    let html = `<h5>Detection Results:</h5>`;

    if (data.detections && data.detections.length > 0) {
        data.detections.forEach(detection => {
            html += `
                <div class="card mb-2">
                    <div class="card-body">
                        <h6 class="card-title">${detection.class}</h6>
                        <p class="card-text">
                            Confidence: ${(detection.confidence * 100).toFixed(1)}%<br>
                            Severity: <span class="badge bg-${getSeverityColor(detection.severity)}">${detection.severity}</span>
                        </p>
                    </div>
                </div>
            `;
        });
    } else {
        html += `<p>No defects detected in this image.</p>`;
    }

    resultsDiv.innerHTML = html;
}

// Helper function to get Bootstrap color based on severity
function getSeverityColor(severity) {
    switch(severity.toLowerCase()) {
        case 'high': return 'danger';
        case 'medium': return 'warning';
        case 'low': return 'success';
        default: return 'secondary';
    }
}

// Show alert notification
function showAlert(message) {
    // Create a notification
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-warning alert-dismissible fade show alert-notification';
    alertDiv.innerHTML = `
        <strong>Alert!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alertDiv);

    // Auto remove after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}