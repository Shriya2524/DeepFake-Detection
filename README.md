# Deepfake Detection System

## Spatiotemporal Analysis + GAN Fingerprinting

A comprehensive deepfake detection system for **B.Tech/BE project** that combines two powerful approaches:

1. **Spatiotemporal Analysis**: Detects temporal inconsistencies in videos using optical flow and LSTM/GRU networks
2. **GAN Fingerprinting**: Identifies GAN-generated artifacts using frequency domain analysis and multi-channel CNNs

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Inference](#inference)
- [Web Interface](#web-interface)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation-results)
- [Contributing](#contributing)

---

## Overview

### Abstract

Deepfake technology poses significant threats to information integrity and personal privacy. This project implements an end-to-end deepfake detection system that analyzes both **spatial** and **temporal** features of videos/images to identify AI-generated content. The system uses:

- **CNN + LSTM architecture** for capturing temporal inconsistencies
- **Multi-channel frequency analysis** for detecting GAN fingerprints
- **Fusion network** for combining both signals into a reliable prediction

### Key Features

- ✅ Video and image deepfake detection
- ✅ Temporal inconsistency analysis using optical flow
- ✅ GAN artifact detection in frequency domain
- ✅ Face detection and extraction
- ✅ Configurable training pipeline
- ✅ CLI and web interface
- ✅ Grad-CAM visualizations (optional)
- ✅ Support for FaceForensics++, Celeb-DF, and custom datasets

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Video or Image                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│   SPATIOTEMPORAL BRANCH         │   │   GAN FINGERPRINT BRANCH        │
│   (Video Analysis)              │   │   (Frame/Image Analysis)        │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ 1. Frame Sampling (10-15 fps)   │   │ 1. Color Space Transform        │
│ 2. Face Detection (MediaPipe)   │   │    (RGB → YCbCr)                │
│ 3. Optical Flow (Farneback)     │   │ 2. Frequency Analysis (DCT)     │
│ 4. CNN Backbone (ResNet18)      │   │ 3. Edge Features (Sobel/Lap.)   │
│ 5. Temporal Model (Bi-LSTM)     │   │ 4. Multi-channel CNN            │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ Output: p_video_fake            │   │ Output: p_image_fake            │
└─────────────────────────────────┘   └─────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FUSION MODULE (MLP)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Features: [p_video, mean(p_frame), std(p_frame), max(p_frame), min(p_frame)]│
│  Output: Final deepfake probability + REAL/DEEPFAKE label                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 16+ and npm
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepfake_detection.git
cd deepfake_detection
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node dependencies**
```bash
npm install
```

3. **Configure environment**
Create a `.env` file in the `frontend/` directory:
```bash
VITE_API_URL=http://localhost:5000
```

---

## Dataset Setup

### Supported Datasets

1. **FaceForensics++** (Recommended)
2. **Celeb-DF**
3. **DFDC (Deepfake Detection Challenge)**
4. **Custom datasets**

### Directory Structure

Place your data in the `data/` folder:

```
data/
├── custom/
│   ├── real/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── fake/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
```

Or for images:
```
data/
├── images/
│   ├── real/
│   │   ├── img1.jpg
│   │   └── ...
│   └── fake/
│       ├── img1.jpg
│       └── ...
```

### Configuration

Edit `config/default_config.yaml`:

```yaml
dataset:
  name: "custom"
  root_path: "data/custom/"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

---

## Training

### Phase 1: Train GAN Fingerprint Model

```bash
python -m src.training.train_gan_fingerprint \
    --config config/default_config.yaml \
    --data-dir data/images \
    --epochs 50 \
    --batch-size 32
```

### Phase 2: Train Spatiotemporal Model

```bash
python -m src.training.train_spatiotemporal \
    --config config/default_config.yaml \
    --data-dir data/custom \
    --epochs 50 \
    --batch-size 8
```

### Phase 3: Train Fusion Model

```bash
python -m src.training.train_fusion \
    --config config/default_config.yaml \
    --spatiotemporal-weights checkpoints/spatiotemporal/best_model.pt \
    --gan-weights checkpoints/gan_fingerprint/best_model.pt \
    --epochs 20
```

### Quick Debug Training

For testing on limited hardware:

```bash
python -m src.training.train_gan_fingerprint \
    --config config/experiment_configs/small_debug.yaml \
    --epochs 5
```

### Training Tips

- Start with smaller batch sizes if you have limited GPU memory
- Use `--device cpu` for CPU-only training
- Monitor training with TensorBoard: `tensorboard --logdir outputs/tensorboard/`

---

## Inference

### CLI - Video Detection

```bash
python -m src.inference.detect_video \
    --video path/to/video.mp4 \
    --spatiotemporal-weights checkpoints/spatiotemporal/best_model.pt \
    --gan-weights checkpoints/gan_fingerprint/best_model.pt \
    --fusion-weights checkpoints/fusion/best_model.pt \
    --threshold 0.5 \
    --output results.json
```

### CLI - Image Detection

```bash
python -m src.inference.detect_image \
    --image path/to/image.jpg \
    --gan-weights checkpoints/gan_fingerprint/best_model.pt \
    --threshold 0.5
```

### Python API

```python
from src.inference.pipeline import DeepfakeDetectionPipeline

# Initialize pipeline
pipeline = DeepfakeDetectionPipeline(
    config_path="config/default_config.yaml",
    device="auto"
)

# Load model weights
pipeline.load_weights(
    spatiotemporal="checkpoints/spatiotemporal/best_model.pt",
    gan_fingerprint="checkpoints/gan_fingerprint/best_model.pt",
    fusion="checkpoints/fusion/best_model.pt"
)

# Detect video
result = pipeline.detect_video("path/to/video.mp4")
print(f"Label: {result['label']}")
print(f"Probability: {result['probability']:.2%}")

# Detect image
result = pipeline.detect_image("path/to/image.jpg")
print(f"Label: {result['label']}")
```

---

## Web Interface

### Launch the Application

**Backend (Flask Server)**
```bash
cd Photo_Detector
python app.py
```
The Flask API will be available at `http://localhost:5000`

**Frontend (React Dashboard)**

In a new terminal:
```bash
cd frontend
npm run dev
```
The web interface will be available at `http://localhost:5173`

### Features

- 📹 Upload and analyze videos
- 🖼️ Upload and analyze images
- 📊 View detection confidence scores
- 📈 Real-time detection results
- 🎨 Modern, responsive UI with dark mode
- 📋 Detailed analysis reports
- 🌐 Upload by file or URL

---

## Project Structure

```
deepfake_detection/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── config/
│   ├── default_config.yaml       # Main configuration
│   └── experiment_configs/       # Experiment-specific configs
├── data/                         # Dataset folder (gitignored)
├── checkpoints/                  # Saved models (gitignored)
├── outputs/                      # Training outputs (gitignored)
├── src/
│   ├── preprocessing/            # Video, face, optical flow processing
│   ├── datasets/                 # PyTorch Dataset classes
│   ├── models/                   # Neural network architectures
│   │   ├── spatiotemporal_model.py
│   │   ├── gan_fingerprint_model.py
│   │   └── fusion_model.py
│   ├── training/                 # Training scripts
│   ├── inference/                # Inference pipeline and CLI
│   ├── evaluation/               # Metrics and visualization
│   ├── api/                      # Flask API endpoints
│   └── utils/                    # Config, logging, device utils
├── Photo_Detector/
│   ├── app.py                    # Flask backend server
│   ├── config.py                 # Backend configuration
│   ├── multi_check_detector.py   # Detection logic
│   ├── static/                   # Static assets
│   └── templates/                # HTML templates
├── frontend/
│   ├── src/                      # React components
│   ├── package.json              # Node dependencies
│   ├── vite.config.ts            # Vite configuration
│   └── index.html                # Entry HTML
├── tests/                        # Unit tests
└── notebooks/                    # Jupyter notebooks (optional)
```

---

## API Reference

### Models

#### SpatiotemporalModel

```python
from src.models import SpatiotemporalModel

model = SpatiotemporalModel(
    backbone="resnet18",
    temporal_type="lstm",
    hidden_size=256,
    num_layers=2,
    use_optical_flow=True
)

output = model(frames, flow)  # frames: (B, T, C, H, W)
# output['probs']: (B, 2) - probability distribution
```

#### GANFingerprintModel

```python
from src.models import GANFingerprintModel

model = GANFingerprintModel(
    backbone="efficientnet_b0",
    use_frequency_branch=True,
    use_edge_branch=True
)

output = model(multichannel_input)  # (B, 9, H, W)
# output['probs']: (B, 2)
```

#### FusionModel

```python
from src.models import FusionModel

model = FusionModel(
    input_features=5,
    hidden_dims=[32, 16]
)

output = model(video_prob, frame_probs)
# output['probs']: (B, 2) - final prediction
```

### Preprocessing

```python
from src.preprocessing import (
    VideoReader,
    FaceExtractor,
    OpticalFlowExtractor,
    FrequencyAnalyzer
)

# Read video
reader = VideoReader(target_fps=10, max_frames=32)
frames = reader.read("video.mp4")

# Extract faces
extractor = FaceExtractor(target_size=(224, 224))
faces = extractor.extract_faces(frame)

# Compute optical flow
flow_extractor = OpticalFlowExtractor()
flows = flow_extractor.compute_sequence(frames)

# Frequency analysis
analyzer = FrequencyAnalyzer()
dct = analyzer.compute_dct(image)
```

---

## Evaluation Results

### Metrics

The system computes the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |

### Expected Performance

On FaceForensics++ (c23 compression):

| Model | Accuracy | AUC |
|-------|----------|-----|
| Spatiotemporal only | ~85% | ~0.90 |
| GAN Fingerprint only | ~88% | ~0.92 |
| Fusion (combined) | ~92% | ~0.95 |

*Note: Results may vary based on training data and hyperparameters.*

---

## Hardware Requirements

### Minimum (CPU Training)

- CPU: 4+ cores
- RAM: 8GB
- Storage: 20GB free

### Recommended (GPU Training)

- GPU: NVIDIA GTX 1080 or better (8GB+ VRAM)
- RAM: 16GB
- Storage: 50GB+ for datasets

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python -m src.training.train_spatiotemporal --batch-size 4
```

**2. MediaPipe Face Detection Fails**
```python
# The system falls back to OpenCV Haar cascade automatically
# Or use center crop: extract_faces=False
```

**3. Video Loading Errors**
```bash
# Ensure ffmpeg is installed
pip install imageio-ffmpeg
```

---

## License

This project is for educational purposes only. Do not use for creating or distributing deepfakes.

---

## Acknowledgments

- [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset
- [MediaPipe](https://mediapipe.dev/) for face detection
- [PyTorch](https://pytorch.org/) deep learning framework
- [Gradio](https://gradio.app/) for web interface

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{deepfake_detection_2025,
  title={Deepfake Detection Using Spatiotemporal Analysis and GAN Fingerprinting},
  author={Shriya R},
  year={2025},
  note={B.Tech/BE Project}
}
```
---

## Thank You!


