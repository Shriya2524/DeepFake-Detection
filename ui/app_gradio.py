"""
Gradio web interface for deepfake detection.

Run with:
    python -m ui.app_gradio

Or:
    cd deepfake_detection && python ui/app_gradio.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tempfile
from typing import Optional, Tuple

import gradio as gr
import numpy as np

# Lazy imports to handle missing dependencies gracefully
try:
    from src.inference.pipeline import DeepfakeDetectionPipeline
    from src.evaluation.visualize_results import plot_frame_scores
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Global pipeline instance
pipeline: Optional[DeepfakeDetectionPipeline] = None
current_model: str = "Full Pipeline (Recommended)"  # Track which model is loaded

# Available models configuration
AVAILABLE_MODELS = {
    "Full Pipeline (Recommended)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": "checkpoints/spatiotemporal/best_model.pt",
        "fusion_path": "checkpoints/fusion/best_model.pt",
        "description": "Complete detection: Spatiotemporal + GAN + Fusion - 98.33% accuracy"
    },
    "Spatiotemporal + GAN (No Fusion)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": "checkpoints/spatiotemporal/best_model.pt",
        "fusion_path": None,
        "description": "Temporal analysis + GAN fingerprint without fusion head - 98.33% accuracy"
    },
    "StyleGAN (Original)": {
        "gan_path": "checkpoints/gan_fingerprint/best_model.pt",
        "spatiotemporal_path": None,
        "fusion_path": None,
        "description": "GAN fingerprint only (StyleGAN) - 94.97% accuracy"
    },
    "PGGAN + StyleGAN (Combined)": {
        "gan_path": "checkpoints/gan_fingerprint_pggan/best_model.pt",
        "spatiotemporal_path": None,
        "fusion_path": None,
        "description": "GAN fingerprint only (PGGAN+StyleGAN) - 100% accuracy"
    }
}


def initialize_pipeline(
    spatiotemporal_weights: str = None,
    gan_weights: str = None,
    fusion_weights: str = None,
    config_path: str = "config/default_config.yaml"
) -> str:
    """Initialize the detection pipeline with model weights."""
    global pipeline, current_model
    
    if not IMPORTS_AVAILABLE:
        return f"❌ Error: Dependencies not available. {IMPORT_ERROR}"
    
    try:
        # Check if config exists
        if not Path(config_path).exists():
            config_path = None
        
        pipeline = DeepfakeDetectionPipeline(
            config_path=config_path,
            device="auto"
        )
        
        # Load weights if provided
        pipeline.load_weights(
            spatiotemporal=spatiotemporal_weights if spatiotemporal_weights else None,
            gan_fingerprint=gan_weights if gan_weights else None,
            fusion=fusion_weights if fusion_weights else None
        )
        
        return "✅ Pipeline initialized successfully!"
    except Exception as e:
        return f"❌ Error initializing pipeline: {str(e)}"


def switch_model(model_name: str) -> str:
    """Switch between different trained models."""
    global pipeline, current_model
    
    if model_name not in AVAILABLE_MODELS:
        return f"❌ Unknown model: {model_name}"
    
    model_info = AVAILABLE_MODELS[model_name]
    gan_path = Path(project_root) / model_info.get("gan_path", "")
    spatiotemporal_path = model_info.get("spatiotemporal_path")
    fusion_path = model_info.get("fusion_path")
    
    if spatiotemporal_path:
        spatiotemporal_path = Path(project_root) / spatiotemporal_path
    if fusion_path:
        fusion_path = Path(project_root) / fusion_path
    
    if not gan_path.exists():
        return f"❌ GAN model not found: {gan_path}\nPlease train the model first."
    
    try:
        # Reinitialize pipeline with new model
        if pipeline is None:
            init_msg = initialize_pipeline()
            if "Error" in init_msg:
                return init_msg
        
        # Load the new weights
        pipeline.load_weights(
            gan_fingerprint=str(gan_path),
            spatiotemporal=str(spatiotemporal_path) if spatiotemporal_path and spatiotemporal_path.exists() else None,
            fusion=str(fusion_path) if fusion_path and fusion_path.exists() else None
        )
        current_model = model_name
        
        return f"✅ Switched to **{model_name}**\n\n{model_info['description']}"
    except Exception as e:
        return f"❌ Error switching model: {str(e)}"


def detect_video(
    video_file,
    threshold: float = 0.5
) -> Tuple[str, str, Optional[object]]:
    """
    Detect deepfake in uploaded video.
    
    Returns:
        Tuple of (result_text, details_text, frame_scores_plot)
    """
    global pipeline
    
    if pipeline is None:
        # Try to initialize with default settings
        init_msg = initialize_pipeline()
        if "Error" in init_msg:
            return init_msg, "", None
    
    if video_file is None:
        return "Please upload a video file.", "", None
    
    try:
        # Set threshold
        pipeline.set_threshold(threshold)
        
        # Run detection
        result = pipeline.detect_video(video_file, return_frame_scores=True)
        
        # Format result
        label = result['label']
        prob = result['probability']
        confidence = result['confidence']
        
        if label == "DEEPFAKE":
            emoji = "🚨"
            color_class = "red"
        else:
            emoji = "✅"
            color_class = "green"
        
        result_text = f"""
## {emoji} Detection Result: **{label}**

**Fake Probability:** {prob:.2%}

**Confidence:** {confidence:.2%}
"""
        
        details_text = f"""
### Detailed Scores

| Metric | Value |
|--------|-------|
| Video (Spatiotemporal) Score | {result['video_score']:.4f} |
| Mean Frame (GAN) Score | {result['mean_frame_score']:.4f} |
| Final Probability | {prob:.4f} |
| Threshold | {threshold:.2f} |
"""
        
        # Create frame scores plot
        frame_plot = None
        if 'frame_scores' in result:
            import matplotlib.pyplot as plt
            
            frame_scores = np.array(result['frame_scores'])
            attention_scores = result.get('attention_scores')
            if attention_scores is not None:
                attention_scores = np.array(attention_scores)
            
            fig = plot_frame_scores(
                frame_scores,
                attention_scores,
                threshold=threshold,
                title="Frame-Level Analysis"
            )
            frame_plot = fig
        
        return result_text, details_text, frame_plot
        
    except Exception as e:
        return f"❌ Error during detection: {str(e)}", "", None


def detect_image(
    image_file,
    threshold: float = 0.5
) -> Tuple[str, str]:
    """
    Detect deepfake in uploaded image.
    
    Returns:
        Tuple of (result_text, details_text)
    """
    global pipeline
    
    if pipeline is None:
        init_msg = initialize_pipeline()
        if "Error" in init_msg:
            return init_msg, ""
    
    if image_file is None:
        return "Please upload an image file.", ""
    
    try:
        # Set threshold
        pipeline.set_threshold(threshold)
        
        # Run detection
        result = pipeline.detect_image(image_file)
        
        # Format result
        label = result['label']
        prob = result['probability']
        confidence = result['confidence']
        
        if label == "DEEPFAKE":
            emoji = "🚨"
        else:
            emoji = "✅"
        
        result_text = f"""
## {emoji} Detection Result: **{label}**

**Fake Probability:** {prob:.2%}

**Confidence:** {confidence:.2%}
"""
        
        details_text = f"""
### Analysis Details

| Metric | Value |
|--------|-------|
| GAN Fingerprint Score | {prob:.4f} |
| Threshold | {threshold:.2f} |

*Note: Image detection uses only the GAN Fingerprint model. For more accurate results on video frames, use video detection.*
"""
        
        return result_text, details_text
        
    except Exception as e:
        return f"❌ Error during detection: {str(e)}", ""


def create_ui():
    """Create the Gradio interface."""
    
    # Custom CSS for styling
    custom_css = """
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .green-result {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .red-result {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    """
    
    with gr.Blocks(
        title="Deepfake Detection System"
    ) as app:
        
        # Header
        gr.Markdown("""
# 🔍 Deepfake Detection System
        
**Spatiotemporal Analysis + GAN Fingerprinting**

This system detects deepfake videos and images using:
- **Spatiotemporal Analysis**: Detects temporal inconsistencies in videos
- **GAN Fingerprinting**: Identifies GAN-generated artifacts in images

Upload a video or image to analyze it for potential deepfakes.
        """)
        
        # Model Switcher (prominent)
        with gr.Group():
            gr.Markdown("### 🧠 Select Detection Model")
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="Full Pipeline (Recommended)",
                    label="Active Model",
                    interactive=True,
                    scale=2
                )
                switch_btn = gr.Button("🔄 Switch Model", variant="primary", scale=1)
            model_status = gr.Markdown(f"**Current:** Full Pipeline (Recommended) - Complete detection system")
            
            switch_btn.click(
                fn=switch_model,
                inputs=[model_selector],
                outputs=model_status
            )
        
        # Model Configuration (collapsible)
        with gr.Accordion("⚙️ Model Configuration", open=False):
            gr.Markdown("*Optional: Load custom model weights*")
            
            with gr.Row():
                spatiotemporal_path = gr.Textbox(
                    label="Spatiotemporal Model Weights",
                    placeholder="checkpoints/spatiotemporal/best_model.pt"
                )
                gan_path = gr.Textbox(
                    label="GAN Fingerprint Model Weights",
                    placeholder="checkpoints/gan_fingerprint/best_model.pt"
                )
                fusion_path = gr.Textbox(
                    label="Fusion Model Weights",
                    placeholder="checkpoints/fusion/best_model.pt"
                )
            
            init_btn = gr.Button("Initialize Pipeline", variant="secondary")
            init_status = gr.Markdown()
            
            init_btn.click(
                fn=initialize_pipeline,
                inputs=[spatiotemporal_path, gan_path, fusion_path],
                outputs=init_status
            )
        
        # Main tabs for video and image detection
        with gr.Tabs():
            
            # Video Detection Tab
            with gr.TabItem("🎬 Video Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        video_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Detection Threshold"
                        )
                        video_btn = gr.Button("🔍 Analyze Video", variant="primary")
                    
                    with gr.Column(scale=1):
                        video_result = gr.Markdown(label="Result")
                        video_details = gr.Markdown(label="Details")
                        video_plot = gr.Plot(label="Frame Analysis")
                
                video_btn.click(
                    fn=detect_video,
                    inputs=[video_input, video_threshold],
                    outputs=[video_result, video_details, video_plot]
                )
            
            # Image Detection Tab
            with gr.TabItem("🖼️ Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Image", type="filepath")
                        image_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Detection Threshold"
                        )
                        image_btn = gr.Button("🔍 Analyze Image", variant="primary")
                    
                    with gr.Column(scale=1):
                        image_result = gr.Markdown(label="Result")
                        image_details = gr.Markdown(label="Details")
                
                image_btn.click(
                    fn=detect_image,
                    inputs=[image_input, image_threshold],
                    outputs=[image_result, image_details]
                )
        
        # Footer
        gr.Markdown("""
---
### 📋 How to Use

1. **For Videos**: Upload a video file (MP4, AVI, MOV). The system will analyze temporal patterns and GAN artifacts across frames.

2. **For Images**: Upload an image file (JPG, PNG). The system will analyze GAN fingerprints in the image.

3. **Adjust Threshold**: Lower threshold = more sensitive (more false positives), Higher threshold = more strict (may miss some deepfakes).

### ⚠️ Disclaimer

This is a research/educational tool. Results should be verified and not used as the sole basis for important decisions.
        """)
    
    return app


def main():
    """Run the Gradio app."""
    # Try to auto-initialize
    print("Initializing Deepfake Detection System...")
    
    # Check for model weights
    checkpoints_dir = project_root / "checkpoints"
    
    spatiotemporal_weights = None
    gan_weights = None
    fusion_weights = None
    
    if (checkpoints_dir / "spatiotemporal" / "best_model.pt").exists():
        spatiotemporal_weights = str(checkpoints_dir / "spatiotemporal" / "best_model.pt")
    
    if (checkpoints_dir / "gan_fingerprint" / "best_model.pt").exists():
        gan_weights = str(checkpoints_dir / "gan_fingerprint" / "best_model.pt")
    
    if (checkpoints_dir / "fusion" / "best_model.pt").exists():
        fusion_weights = str(checkpoints_dir / "fusion" / "best_model.pt")
    
    # Initialize pipeline
    if IMPORTS_AVAILABLE:
        init_msg = initialize_pipeline(
            spatiotemporal_weights=spatiotemporal_weights,
            gan_weights=gan_weights,
            fusion_weights=fusion_weights
        )
        print(init_msg)
    else:
        print(f"Warning: Some dependencies not available. {IMPORT_ERROR}")
        print("The UI will still load, but detection may not work.")
    
    # Create and launch app
    app = create_ui()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()

