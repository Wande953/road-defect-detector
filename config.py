import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
    UPLOAD_FOLDER = 'uploads'
    REPORTS_FOLDER = 'reports'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

    CLASS_NAMES = ['Alligator', 'Longitudinal', 'Pothole', 'Transverse']

    SEVERITY_LEVELS = {
        'Pothole': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Alligator': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Longitudinal': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
        'Transverse': {'low': 0.3, 'medium': 0.6, 'high': 0.8}
    }

    CONFIDENCE_THRESHOLD = 0.2


# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
os.makedirs('static/maps', exist_ok=True)