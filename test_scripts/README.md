# Deepfake Detection System - Test Scripts

This folder contains comprehensive test scripts for validating all components of the deepfake detection system.

## 📁 Folder Structure

```
test_scripts/
├── run_all_tests.py              # Master test runner
├── README.md                     # This file
├── database_testing/
│   └── test_database.py          # Dataset validation tests
├── api_testing/
│   └── test_api_endpoints.py     # API endpoint tests
├── model_testing/
│   └── test_models.py            # Model accuracy tests
├── frontend_testing/
│   └── test_frontend.py          # Frontend & PDF tests
├── dashboard_testing/
│   └── test_dashboard_output.py  # Dashboard output tests
├── video_detection_testing/
│   └── test_video_detection.py   # Video detection tests
├── image_detection_testing/
│   └── test_image_detection.py   # Image detection tests
└── accuracy_graphs/
    ├── test_accuracy_graphs.py   # Accuracy graph generation
    └── outputs/                  # Generated graphs
```

## 🚀 Quick Start

### Run All Tests
```bash
cd test_scripts
python run_all_tests.py
```

### Run Specific Tests
```bash
# Run only database tests
python run_all_tests.py database

# Run multiple specific tests
python run_all_tests.py database api models

# List available tests
python run_all_tests.py --list
```

### Run Individual Test Scripts
```bash
# Database testing
python database_testing/test_database.py

# API testing (requires server running)
python api_testing/test_api_endpoints.py

# Model testing
python model_testing/test_models.py
```

## 📋 Test Categories

### 1. Database Testing (`database_testing/`)
Tests dataset validation and input-output verification:
- Dataset directory structure verification
- Data utility functions (splits, balancing, stats)
- ImageDataset class functionality
- VideoDataset class functionality
- DataLoader batch processing
- Checkpoint file verification

### 2. API Endpoints Testing (`api_testing/`)
Tests all API endpoints:
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /models/switch` - Switch active model
- `POST /detect/image` - Image detection
- `POST /detect/video` - Video detection
- Error handling validation

**Note:** Requires the API server to be running on `localhost:8000`

### 3. Model Testing (`model_testing/`)
Tests model accuracy and functionality:
- Spatiotemporal Model creation and inference
- GAN Fingerprint Model creation and inference
- Fusion Model creation and inference
- GAN Family Classifier
- Accuracy metrics simulation

### 4. Frontend & PDF Testing (`frontend_testing/`)
Tests frontend components and PDF generation:
- Frontend directory structure
- package.json configuration
- React components verification
- App.tsx structure analysis
- PDF report generation capability
- Styling files verification

### 5. Dashboard Output Testing (`dashboard_testing/`)
Tests dashboard output components:
- Deepfake/Real label output
- Confidence score calculation
- GAN signal strength measurement
- GAN family prediction
- Complete dashboard output simulation

### 6. Video Detection Testing (`video_detection_testing/`)
Tests video detection output:
- Frame-level suspicious regions detection
- Spatiotemporal score calculation
- Fusion score computation
- Attention-based frame analysis
- Complete video detection output

### 7. Image Detection Testing (`image_detection_testing/`)
Tests image detection output:
- GAN fingerprint result validation
- GAN family classification
- Multi-scale feature analysis
- Confidence calibration
- Complete image detection output

### 8. Accuracy Graphs (`accuracy_graphs/`)
Generates accuracy visualizations:
- Training/validation curves
- Model comparison bar charts
- ROC curves
- Confusion matrices
- Per-class accuracy charts
- Comprehensive accuracy report

## 📊 Output Examples

### Terminal Output
Each test produces formatted terminal output with:
- ✓ Pass/Fail indicators
- Colored status messages
- Detailed metrics
- Visual representations (ASCII charts)

### Generated Files
The `accuracy_graphs/outputs/` folder contains:
- `training_curves.png`
- `model_comparison.png`
- `roc_curves.png`
- `confusion_matrices.png`
- `per_class_accuracy.png`
- `comprehensive_report.png`

## 🔧 Requirements

### Python Dependencies
```bash
pip install colorama numpy torch requests
# Optional for graphs:
pip install matplotlib scikit-learn
```

### For API Testing
Start the API server before running API tests:
```bash
python -m src.api.main
```

## 📈 Expected Accuracies

| Model | Accuracy |
|-------|----------|
| GAN Fingerprint | 94.97% |
| Spatiotemporal | 96.00% |
| Fusion | 98.33% |
| PGGAN+StyleGAN | 100.00% |

## ⚠️ Notes

1. **API Tests**: Require the Flask server to be running
2. **Model Tests**: May use GPU if available (CUDA)
3. **Graph Generation**: Requires matplotlib for PNG output
4. **Test Data**: Some tests use sample data from the project

## 🐛 Troubleshooting

### "Connection refused" in API tests
- Make sure the API server is running: `python -m src.api.main`

### "Module not found" errors
- Run from the project root or add project to PYTHONPATH

### No graphs generated
- Install matplotlib: `pip install matplotlib`

### CUDA errors
- Tests will fall back to CPU if CUDA is not available

## 📝 License

Part of the Deepfake Detection System project.
