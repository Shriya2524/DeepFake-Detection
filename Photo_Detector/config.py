"""
Configuration settings for AI Image Detector
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model settings
MODEL_PATH = os.getenv('MODEL_PATH', 'image_classifier_model.keras')
IMAGE_SIZE = (128, 128)  # Expected input size for the model

# Upload settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB (increased for large images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
UPLOAD_FOLDER = BASE_DIR / 'uploads'

# URL download settings
DOWNLOAD_TIMEOUT = 30  # seconds (increased for large files)
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Prediction settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to show result
CLASS_NAMES = ['AI_generated', 'Natural']

# Security settings
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
RATE_LIMIT_REQUESTS = 100  # requests per hour per IP

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'logs' / 'app.log'

# Flask settings
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_size(file_path):
    """Get file size in bytes"""
    return os.path.getsize(file_path)
