"""
Flask Web Application for AI Image Detector
Provides web interface and API for image classification
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras import models
import requests
from PIL import Image
from io import BytesIO
from loguru import logger
import config
from multi_check_detector import MultiCheckDetector


app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = str(config.UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# Enable CORS for API endpoints
CORS(app)

# Configure logging
logger.add(
    config.LOG_FILE,
    rotation="10 MB",
    retention="7 days",
    level=config.LOG_LEVEL
)

# Load model at startup
try:
    model = models.load_model(config.MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from {config.MODEL_PATH}")
    # Initialize multi-check detector
    detector = MultiCheckDetector(model)
    logger.info("✅ Multi-check detector initialized")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None
    detector = None


def preprocess_image(img_array):
    """Preprocess image for model prediction"""
    try:
        # Handle different color formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # Resize to model's expected input
        img_array = cv2.resize(img_array, config.IMAGE_SIZE)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise


def predict_image(img_array):
    """Make comprehensive multi-check prediction"""
    if detector is None:
        raise RuntimeError("Detector not loaded")
    
    try:
        # Run comprehensive analysis with multiple checks
        result = detector.analyze_comprehensive(img_array)
        
        return {
            'prediction': result['final_prediction'],
            'confidence': result['confidence'],
            'ai_probability': result['ai_probability'],
            'natural_probability': result['natural_probability'],
            'final_score': result['final_score'],
            'votes': result['votes'],
            'individual_checks': result['individual_checks'],
            'analysis_summary': result['analysis_summary']
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Handle file upload prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not config.allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Make prediction
        result = predict_image(img)
        result['source'] = 'upload'
        result['filename'] = secure_filename(file.filename)
        
        logger.info(f"Upload prediction: {file.filename} -> {result['prediction']} ({result['confidence']:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/url', methods=['POST'])
def predict_url():
    """Handle URL-based prediction"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        image_url = data['url']
        
        # Validate URL
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Download image
        headers = {'User-Agent': config.USER_AGENT}
        response = requests.get(
            image_url,
            timeout=config.DOWNLOAD_TIMEOUT,
            headers=headers,
            stream=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return jsonify({
                'error': f'URL does not point to an image (Content-Type: {content_type})'
            }), 400
        
        # Check size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > config.MAX_DOWNLOAD_SIZE:
            return jsonify({'error': f'Image too large (max {config.MAX_DOWNLOAD_SIZE // (1024*1024)}MB)'}), 400
        
        # Open image
        img = Image.open(BytesIO(response.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array and BGR
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Make prediction
        result = predict_image(img_array)
        result['source'] = 'url'
        result['url'] = image_url
        
        logger.info(f"URL prediction: {image_url} -> {result['prediction']} ({result['confidence']:.2%})")
        
        return jsonify(result)
    
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 408
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Connection error - check URL'}), 500
    except requests.exceptions.HTTPError as e:
        return jsonify({'error': f'HTTP Error: {e}'}), 500
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': config.MODEL_PATH
    })


@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': f'File too large (max {config.MAX_FILE_SIZE // (1024*1024)}MB)'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    if model is None:
        print("❌ ERROR: Model could not be loaded. Please check MODEL_PATH in config.py")
        exit(1)
    
    print("="*60)
    print("🚀 AI Image Detector Web Server")
    print("="*60)
    print(f"📍 Running on: http://{config.HOST}:{config.PORT}")
    print(f"🧠 Model: {config.MODEL_PATH}")
    print(f"📊 Debug mode: {config.DEBUG}")
    print("="*60)
    print("\nEndpoints:")
    print("  • Home:         GET  /")
    print("  • Upload:       POST /api/predict/upload")
    print("  • URL:          POST /api/predict/url")
    print("  • Health Check: GET  /api/health")
    print("="*60)
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
