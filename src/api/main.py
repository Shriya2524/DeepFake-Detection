import base64
import os
import sys
import tempfile
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
from flask_jwt_extended.exceptions import NoAuthorizationError

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import DeepfakeDetectionPipeline
from .models import db, AnalysisHistory
from .auth import auth_bp
from .history import history_bp
from .ai_generated.multi_check_detector import MultiCheckDetector

# Keras model for AI-Generated detection
import numpy as np
try:
    from keras import models as keras_models
    AI_GENERATED_MODEL_PATH = Path(__file__).parent / "ai_generated" / "image_classifier_model.keras"
    ai_generated_model = None
    ai_generated_detector = None
except ImportError:
    keras_models = None
    ai_generated_model = None
    ai_generated_detector = None

app = Flask(__name__)

# CORS configuration - allow Railway, dev tunnels and localhost
CORS(app, 
     origins=[
         "http://localhost:3000",
         "http://127.0.0.1:3000",
         "https://vjb8z6xc-3000.inc1.devtunnels.ms",
         "https://vjb8z6xc-8000.inc1.devtunnels.ms",
         "https://frontend-production-7049.up.railway.app",
     ],
     resources={r"/*": {"origins": "*"}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

# Database configuration
db_path = project_root / "src" / "api" / "deepfake.db"
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'deepfake-detection-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)  # Token expires in 7 days

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(history_bp)

# Create database tables
with app.app_context():
    db.create_all()

# Global state
pipeline: Optional[DeepfakeDetectionPipeline] = None
current_model: Optional[str] = None
DEFAULT_MODEL = "Full Pipeline (Recommended)"

AVAILABLE_MODELS = {
    "Full Pipeline (Recommended)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": "checkpoints/spatiotemporal/best_model.pt",
        "fusion_path": "checkpoints/fusion/best_model.pt",
        "gan_family_path": "checkpoints/gan_fingerprint_family/best_model.pt",
        "description": "Complete detection: All 3 models combined - 98.33% accuracy"
    },
    "Spatiotemporal + GAN + Fusion": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": "checkpoints/spatiotemporal/best_model.pt",
        "fusion_path": "checkpoints/fusion/best_model.pt",
        "gan_family_path": "checkpoints/gan_fingerprint_family/best_model.pt",
        "description": "Temporal analysis + GAN fingerprint + Fusion head - 98.33% accuracy"
    },
    "Spatiotemporal + GAN (No Fusion)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": "checkpoints/spatiotemporal/best_model.pt",
        "fusion_path": None,
        "gan_family_path": "checkpoints/gan_fingerprint_family/best_model.pt",
        "description": "Temporal analysis + GAN fingerprint without fusion head - 98.33% accuracy"
    },
    "StyleGAN (Original)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": None,
        "fusion_path": None,
        "gan_family_path": "checkpoints/gan_fingerprint_family/best_model.pt",
        "description": "GAN fingerprint only (StyleGAN) - 94.97% accuracy"
    },
    "PGGAN + StyleGAN (Combined)": {
        "gan_path": "checkpoints/gan_fingerprint_pggan/best_model.pt",
        "spatiotemporal_path": None,
        "fusion_path": None,
        "gan_family_path": "checkpoints/gan_fingerprint_family/best_model.pt",
        "description": "GAN fingerprint only (PGGAN+StyleGAN) - 100% accuracy"
    }
}


def get_current_user_id_optional():
    """Get current user ID if authenticated, otherwise return None."""
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None


def save_analysis_to_history(user_id: int, filename: str, media_type: str, result: str, confidence: float, details: dict = None):
    """Save analysis result to user's history."""
    try:
        record = AnalysisHistory(
            user_id=user_id,
            filename=filename,
            type=media_type.capitalize(),
            result=result,
            confidence=confidence,
            details=details
        )
        db.session.add(record)
        db.session.commit()
        return record
    except Exception as e:
        print(f"Error saving to history: {e}")
        db.session.rollback()
        return None


def initialize_pipeline_if_needed():
    global pipeline
    if pipeline is None:
        try:
            pipeline = DeepfakeDetectionPipeline(device="auto")
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return False
    return True


def initialize_ai_generated_detector():
    """Initialize the AI-Generated image detector."""
    global ai_generated_model, ai_generated_detector
    
    if ai_generated_detector is not None:
        return True
    
    if keras_models is None:
        print("Keras not available for AI-Generated detection")
        return False
    
    try:
        if AI_GENERATED_MODEL_PATH.exists():
            ai_generated_model = keras_models.load_model(str(AI_GENERATED_MODEL_PATH))
            ai_generated_detector = MultiCheckDetector(ai_generated_model)
            print(f"✅ AI-Generated detector loaded from {AI_GENERATED_MODEL_PATH}")
            return True
        else:
            print(f"❌ AI-Generated model not found at {AI_GENERATED_MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading AI-Generated detector: {e}")
        return False


@app.route('/')
def home():
    """API homepage with available endpoints."""
    return jsonify({
        "message": "Deepfake Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "models": "/models",
            "detect_image": "/detect/image",
            "detect_video": "/detect/video",
            "auth": {
                "register": "/auth/register",
                "login": "/auth/login"
            },
            "history": "/history"
        },
        "documentation": "Visit /models to see available detection models"
    })


def ensure_model_loaded(requested_model: Optional[str] = None):
    """
    Lazily load the requested model if not already active.
    
    Returns:
        Tuple (success: bool, active_model: Optional[str], error: Optional[str])
    """
    global pipeline, current_model
    
    if not initialize_pipeline_if_needed():
        return False, None, "Pipeline initialization failed"
    
    target_model = requested_model or current_model or DEFAULT_MODEL
    
    if target_model not in AVAILABLE_MODELS:
        return False, None, f"Unknown model: {target_model}"
    
    if current_model == target_model:
        return True, current_model, None
    
    model_info = AVAILABLE_MODELS[target_model]
    gan_rel_path = model_info.get("gan_path") or model_info.get("path")
    spatiotemporal_rel = model_info.get("spatiotemporal_path")
    fusion_rel = model_info.get("fusion_path")
    gan_family_rel = model_info.get("gan_family_path")
    
    if not gan_rel_path:
        return False, None, f"No GAN weights configured for model {target_model}"
    
    gan_path = project_root / gan_rel_path
    spatiotemporal_path = project_root / spatiotemporal_rel if spatiotemporal_rel else None
    fusion_path = project_root / fusion_rel if fusion_rel else None
    gan_family_path = project_root / gan_family_rel if gan_family_rel else None
    
    if not gan_path.exists():
        return False, None, f"Model weights not found at {gan_path}"
    
    # Check if GAN family weights exist (optional - don't fail if missing)
    gan_family_available = gan_family_path and gan_family_path.exists()
    if gan_family_path and not gan_family_available:
        print(f"GAN family weights not found at {gan_family_path} - family detection will be disabled")
    
    try:
        pipeline.load_weights(
            gan_fingerprint=str(gan_path),
            spatiotemporal=str(spatiotemporal_path) if spatiotemporal_path else None,
            fusion=str(fusion_path) if fusion_path else None,
            gan_family=str(gan_family_path) if gan_family_available else None,
        )
        current_model = target_model
        return True, current_model, None
    except Exception as e:
        return False, None, str(e)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": current_model or DEFAULT_MODEL,
        "loaded": current_model is not None
    })

@app.route("/models", methods=["GET"])
def list_models():
    active_model = current_model or DEFAULT_MODEL
    models = []
    for name, info in AVAILABLE_MODELS.items():
        gan_rel_path = info.get("gan_path") or info.get("path")
        gan_path = project_root / gan_rel_path if gan_rel_path else None
        gan_family_rel = info.get("gan_family_path")
        gan_family_path = project_root / gan_family_rel if gan_family_rel else None
        models.append({
            "name": name,
            "description": info["description"],
            "active": (name == active_model),
            "loaded": (name == current_model),
            "available": gan_path.exists() if gan_path else False,
            "gan_family_available": gan_family_path.exists() if gan_family_path else False
        })
    return jsonify(models)

@app.route("/models/switch", methods=["POST"])
def switch_model():
    global pipeline, current_model
    
    model_name = request.form.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
        
    success, active_model, error = ensure_model_loaded(model_name)
    if not success:
        status = 404 if error and "not found" in error.lower() else 503 if error and "initialization" in error.lower() else 400
        return jsonify({"error": error or "Unable to switch model"}), status
    
    return jsonify({"status": "success", "message": f"Switched to {active_model}"})

@app.route("/detect/video", methods=["POST"])
def detect_video():
    global pipeline
    
    if not initialize_pipeline_if_needed():
        return jsonify({"error": "Pipeline initialization failed"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    model_name = request.form.get("model_name")
    try:
        threshold = float(request.form.get("threshold", 0.5))
    except ValueError:
        return jsonify({"error": "Invalid threshold value"}), 400
    
    success, active_model, error = ensure_model_loaded(model_name)
    if not success:
        status = (
            404 if error and "not found" in error.lower()
            else 503 if error and "initialization" in error.lower()
            else 400
        )
        return jsonify({"error": error or "Model not available"}), status
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        pipeline.set_threshold(threshold)
        result = pipeline.detect_video(tmp_path, return_frame_scores=True, return_frames=True)
        
        frame_scores = result.get('frame_scores')
        attention_scores = result.get('attention_scores')
        
        frame_scores = result.get('frame_scores')
        frame_scores_list = frame_scores.tolist() if frame_scores is not None else None
        fake_frames = None
        if frame_scores_list:
            fake_frames = sorted(
                [{"index": i + 1, "score": float(s)} for i, s in enumerate(frame_scores_list)],
                key=lambda x: x["score"],
                reverse=True
            )[:5]
            # Attach previews if frames are available
            frames_array = result.get('frames')
            if frames_array is not None:
                previews = []
                for item in fake_frames:
                    idx = item["index"] - 1
                    if 0 <= idx < len(frames_array):
                        frame = frames_array[idx]
                        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if ok:
                            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                            item["preview"] = f"data:image/jpeg;base64,{b64}"
                        previews.append(item)
                fake_frames = fake_frames

        response = {
            "label": result['label'],
            "probability": float(result['probability']),
            "confidence": float(result['confidence']),
            "threshold": float(result.get('threshold', threshold)),
            "model": active_model,
            "details": {
                "video_score": float(result.get('video_score', result['probability'])),
                "mean_frame_score": float(result.get('mean_frame_score', result['probability'])),
                "std_frame_score": float(result.get('std_frame_score', 0.0)),
                "max_frame_score": float(result.get('max_frame_score', 0.0)),
                "min_frame_score": float(result.get('min_frame_score', 0.0)),
                "score_source": result.get('score_source'),
                "gan_family": result.get('gan_family') or "Not available (no family model loaded)",
                "gan_family_confidence": float(result.get('gan_family_confidence', 0.0)) if result.get('gan_family_confidence') else None,
                "family_probs": result.get('family_probs'),
                "fake_frames": fake_frames
            },
            "frame_scores": frame_scores_list,
            "attention_scores": attention_scores.tolist() if attention_scores is not None else None
        }
        
        # Save to history if user is authenticated
        user_id = get_current_user_id_optional()
        if user_id:
            is_deepfake = result['label'] == 'DEEPFAKE'
            save_analysis_to_history(
                user_id=user_id,
                filename=file.filename,
                media_type='Video',
                result='Fake' if is_deepfake else 'Real',
                confidence=float(result['confidence']) * 100,
                details=response.get('details')
            )
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route("/detect/image", methods=["POST"])
def detect_image():
    global pipeline
    
    if not initialize_pipeline_if_needed():
        return jsonify({"error": "Pipeline initialization failed"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    model_name = request.form.get("model_name")
    try:
        threshold = float(request.form.get("threshold", 0.5))
    except ValueError:
        return jsonify({"error": "Invalid threshold value"}), 400
    
    success, active_model, error = ensure_model_loaded(model_name)
    if not success:
        status = (
            404 if error and "not found" in error.lower()
            else 503 if error and "initialization" in error.lower()
            else 400
        )
        return jsonify({"error": error or "Model not available"}), status
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        pipeline.set_threshold(threshold)
        result = pipeline.detect_image(tmp_path)
        
        response = {
            "label": result['label'],
            "probability": float(result['probability']),
            "confidence": float(result['confidence']),
            "threshold": float(result.get('threshold', threshold)),
            "model": active_model,
            "details": {
                "gan_score": float(result.get('gan_score', result['probability'])),
                "mean_frame_score": float(result.get('gan_score', result['probability'])),
                "video_score": None,
                "score_source": result.get('score_source'),
                "gan_family": result.get('gan_family') or "Not available (no family model loaded)",
                "gan_family_confidence": float(result.get('gan_family_confidence', 0.0)) if result.get('gan_family_confidence') else None,
                "family_probs": result.get('family_probs'),
                "fake_frames": None
            },
            "frame_scores": None,
            "attention_scores": None
        }
        
        # Save to history if user is authenticated
        user_id = get_current_user_id_optional()
        if user_id:
            is_deepfake = result['label'] == 'DEEPFAKE'
            save_analysis_to_history(
                user_id=user_id,
                filename=file.filename,
                media_type='Image',
                result='Fake' if is_deepfake else 'Real',
                confidence=float(result['confidence']) * 100,
                details=response.get('details')
            )
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/detect/ai-generated", methods=["POST"])
def detect_ai_generated():
    """
    Detect if an image is AI-generated (DALL-E, Midjourney, Stable Diffusion, etc.)
    vs a natural/real photo.
    """
    global ai_generated_detector
    
    # Initialize detector if not loaded
    if not initialize_ai_generated_detector():
        return jsonify({"error": "AI-Generated detector not available"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Read image with OpenCV
        img = cv2.imread(tmp_path)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Run multi-check analysis
        result = ai_generated_detector.analyze_comprehensive(img)
        
        # Build response
        is_ai_generated = result['final_prediction'] == 'AI_Generated'
        response = {
            "label": "AI_GENERATED" if is_ai_generated else "NATURAL",
            "prediction": result['final_prediction'],
            "probability": result['ai_probability'],
            "confidence": result['confidence'],
            "final_score": result['final_score'],
            "confidence_level": result['confidence_level'],
            "votes": result['votes'],
            "individual_checks": result['individual_checks'],
            "analysis_summary": result['analysis_summary'],
            "filename": file.filename
        }
        
        # Save to history if user is authenticated
        user_id = get_current_user_id_optional()
        if user_id:
            save_analysis_to_history(
                user_id=user_id,
                filename=file.filename,
                media_type='AI-Check',
                result='AI-Generated' if is_ai_generated else 'Natural',
                confidence=float(result['confidence']) * 100,
                details={
                    'final_score': result['final_score'],
                    'votes': result['votes'],
                    'confidence_level': result['confidence_level'],
                    'individual_checks': result['individual_checks']
                }
            )
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/detect/ai-generated/health", methods=["GET"])
def ai_generated_health():
    """Check AI-Generated detector health."""
    return jsonify({
        "status": "healthy" if ai_generated_detector is not None else "not_loaded",
        "model_path": str(AI_GENERATED_MODEL_PATH) if 'AI_GENERATED_MODEL_PATH' in globals() else None,
        "model_exists": AI_GENERATED_MODEL_PATH.exists() if 'AI_GENERATED_MODEL_PATH' in globals() else False
    })


if __name__ == "__main__":
    # Initialize pipeline on startup
    if initialize_pipeline_if_needed():
        ok, _, err = ensure_model_loaded(DEFAULT_MODEL)
        if not ok and err:
            print(f"Startup warning: {err}")
    app.run(host="0.0.0.0", port=8000, debug=False)
