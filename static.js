// static/js/dashboard.js

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    connectWebSocket();
});

function initCharts() {
    // Defect distribution chart
    const defectCtx = document.getElementById('defectChart').getContext('2d');
    const defectChart = new Chart(defectCtx, {
        type: 'pie',
        data: {
            labels: ['Alligator', 'Longitudinal', 'Pothole', 'Transverse'],
            datasets: [{
                data: [0, 0, 0, 0], // Initial empty data
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

    // Severity chart
    const severityCtx = document.getElementById('severityChart').getContext('2d');
    const severityChart = new Chart(severityCtx, {
        type: 'bar',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                label: 'Defects by Severity',
                data: [0, 0, 0], // Initial empty data
                backgroundColor: ['#dc3545', '#ffc107', '#28a745']
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Severity Distribution'
                }
            }
        }
    });

    // Store charts for later updates
    window.defectChart = defectChart;
    window.severityChart = severityChart;
}

function connectWebSocket() {
    // Socket.io is already available globally from your HTML template
    const socket = io();

    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('new_detection', function(data) {
        updateCharts(data.stats);
        updateDetectionTable(data.detections);
    });

    socket.on('initial_data', function(data) {
        updateCharts(data.stats);
        updateDetectionTable(data.history.flatMap(h => h.detections));
    });
}

function updateCharts(stats) {
    // Update defect chart
    const defectLabels = Object.keys(stats.by_class);
    const defectData = Object.values(stats.by_class);

    window.defectChart.data.labels = defectLabels;
    window.defectChart.data.datasets[0].data = defectData;
    window.defectChart.update();

    // Update total detections counter
    document.getElementById('totalDetections').textContent = stats.total_detections;
}

function updateDetectionTable(detections) {
    const tableBody = document.getElementById('detectionsTable');
    tableBody.innerHTML = ''; // Clear existing rows

    // Show latest 10 detections
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

function getSeverityColor(severity) {
    switch(severity.toLowerCase()) {
        case 'high': return 'danger';
        case 'medium': return 'warning';
        case 'low': return 'success';
        default: return 'secondary';
    }
}