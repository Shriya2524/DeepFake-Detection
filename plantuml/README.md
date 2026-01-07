# PlantUML Diagrams for Deepfake Detection Project

This folder contains PlantUML diagram definitions for documenting the testing and output aspects of the deepfake detection system.

## 📁 Folder Structure

```
plantuml/
├── README.md                    # This file
├── all_diagrams.puml           # All diagrams in one file
├── generate_diagrams.py        # Python script to generate PNG images
├── individual/                  # Individual diagram files
│   ├── 01_database_testing.puml
│   ├── 02_api_endpoints_testing.puml
│   ├── 03_model_testing.puml
│   ├── 04_frontend_pdf_testing.puml
│   ├── 05_dashboard_output.puml
│   ├── 06_video_detection_output.puml
│   ├── 07_image_detection_output.puml
│   └── 08_accuracy_graph.puml
└── images/                      # Generated PNG images (after running script)
```

## 📊 Diagrams Included

| # | Diagram | Description |
|---|---------|-------------|
| 1 | Database Testing | Dataset validation, input-output verification |
| 2 | API Endpoints Testing | POST /detect/video, /detect/image, /models, /health |
| 3 | Model Testing | Spatiotemporal Model, GAN Fingerprint Model accuracy tests |
| 4 | Frontend & PDF Testing | Component tests, PDF report validation |
| 5 | Dashboard Output | Deepfake label, confidence, GAN signal, family prediction |
| 6 | Video Detection Output | Frame-level analysis, spatiotemporal & fusion scores |
| 7 | Image Detection Output | GAN fingerprint, family classification |
| 8 | Accuracy Graph | Model comparison charts (GAN, Spatio, Fusion) |

## 🚀 How to Generate Images

### Option 1: Run the Python Script (Recommended)

```bash
cd plantuml
python generate_diagrams.py
```

This will:
- Download PlantUML JAR if Java is available
- Generate PNG images using local JAR or online server
- Save images to `plantuml/images/` folder

### Option 2: Use PlantUML Online Server

1. Go to [PlantUML Web Server](http://www.plantuml.com/plantuml/uml)
2. Copy-paste the content from any `.puml` file
3. Click "Submit" to generate the diagram
4. Right-click and save the image

### Option 3: Use VS Code Extension

1. Install the "PlantUML" extension (jebbs.plantuml)
2. Open any `.puml` file
3. Press `Alt+D` to preview
4. Right-click and export as PNG

### Option 4: Use PlantUML CLI

```bash
# Download PlantUML JAR
curl -L -o plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2024.6/plantuml-1.2024.6.jar

# Generate all diagrams
java -jar plantuml.jar -tpng individual/*.puml

# Generate a single diagram
java -jar plantuml.jar -tpng individual/01_database_testing.puml
```

## 📝 Quick Copy-Paste Reference

For quick use, here are all diagram codes you can directly paste into any PlantUML tool:

### 1. Database Testing
See: `individual/01_database_testing.puml`

### 2. API Endpoints Testing  
See: `individual/02_api_endpoints_testing.puml`

### 3. Model Testing
See: `individual/03_model_testing.puml`

### 4. Frontend & PDF Testing
See: `individual/04_frontend_pdf_testing.puml`

### 5. Dashboard Output
See: `individual/05_dashboard_output.puml`

### 6. Video Detection Output
See: `individual/06_video_detection_output.puml`

### 7. Image Detection Output
See: `individual/07_image_detection_output.puml`

### 8. Accuracy Graph
See: `individual/08_accuracy_graph.puml`

## 🎨 Customization

You can modify the diagrams by:
- Changing colors: Replace `#LightBlue`, `#LightGreen`, etc.
- Updating metrics: Change accuracy values, scores
- Adding elements: Add more cards, rectangles, notes
- Changing theme: Replace `!theme plain` with other themes

### Available Themes
```
!theme plain
!theme cerulean
!theme superhero
!theme united
!theme materia
!theme sketchy-outline
```

## 🔗 Useful Links

- [PlantUML Official](https://plantuml.com/)
- [PlantUML Web Server](http://www.plantuml.com/plantuml/uml)
- [VS Code PlantUML Extension](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml)
- [PlantUML Themes](https://plantuml.com/theme)
