# 🚀 AI Image Detector - Quick Start Guide

## What's New? 🎉

Your project has been **completely enhanced** with:

1. **✅ URL Support** - Grab images directly from the internet
2. **✅ Web Interface** - Beautiful drag-and-drop UI
3. **✅ Multiple Input Methods** - File upload or URL
4. **✅ Enhanced CLI Tool** - Interactive mode & batch processing
5. **✅ Production-Ready** - Proper error handling, logging, config

---

## 🎯 Quick Start

### Option 1: Command Line (Enhanced Detector)

**Single Image (Local):**
```powershell
python enhanced_detector.py duck.jpg
```

**Single Image (From Internet URL):**
```powershell
python enhanced_detector.py https://example.com/image.jpg
```

**Interactive Mode:**
```powershell
python enhanced_detector.py
# Then paste URLs or file paths when prompted
```

**Batch Mode:**
```powershell
python enhanced_detector.py --batch duck.jpg aiimag.jpg https://url.com/image.jpg
```

---

### Option 2: Web Interface (Recommended)

**Start the Web Server:**
```powershell
python app.py
```

**Then open your browser:**
```
http://localhost:5000
```

**Features:**
- 📤 Drag & drop images
- 🌐 Paste URLs from internet
- 📊 Beautiful results with confidence scores
- 📱 Mobile-friendly responsive design

---

## 📁 Project Structure

```
AI_Generated_photo_Detector-main/
├── app.py                          # Flask web application
├── enhanced_detector.py            # Enhanced CLI tool (URL support)
├── train_model.py                  # Training script (renamed from tarin_model.py)
├── test_model.py                   # Simple testing script
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── image_classifier_model.keras    # Trained model
├── templates/
│   └── index.html                  # Web UI
├── static/
│   └── style.css                   # Styling
└── README.md                       # Original documentation
```

---

## 🔧 Configuration

Edit `config.py` to customize:

- `MAX_FILE_SIZE` - Maximum upload size (default: 10MB)
- `ALLOWED_EXTENSIONS` - Supported image formats
- `DOWNLOAD_TIMEOUT` - URL download timeout (default: 15s)
- `IMAGE_SIZE` - Model input size (128x128)
- `PORT` - Web server port (default: 5000)

---

## 📝 Examples

### CLI Examples:

```powershell
# Local file
python enhanced_detector.py duck.jpg

# Internet URL
python enhanced_detector.py https://picsum.photos/200

# Multiple files
python enhanced_detector.py --batch image1.jpg image2.jpg https://url.com/img.jpg

# Interactive mode (keeps asking for input)
python enhanced_detector.py
```

### Web API Examples:

**Health Check:**
```bash
GET http://localhost:5000/api/health
```

**Upload Prediction:**
```bash
POST http://localhost:5000/api/predict/upload
Body: multipart/form-data with 'file' field
```

**URL Prediction:**
```bash
POST http://localhost:5000/api/predict/url
Body: {"url": "https://example.com/image.jpg"}
```

---

## 🐛 Known Issues

**NumPy Warning Messages:**
You may see warnings about NumPy 1.x vs 2.x compatibility. These are **harmless** and don't affect functionality. The model works perfectly despite these warnings.

---

## 🚀 Next Steps

### Improve Model Accuracy:
1. Collect more training data (AI-generated & natural images)
2. Implement transfer learning (see commented suggestions in code)
3. Train for more epochs with data augmentation

### Deploy Online:
1. Use Railway, Heroku, or Vercel for hosting
2. Set environment variables for production
3. Use proper SECRET_KEY in config.py

### Add Features:
- Batch URL processing
- Image history/database
- User accounts
- API rate limiting
- Confidence threshold settings

---

## 💡 Tips

- **Best results:** Use clear, high-quality images
- **Supported formats:** JPG, PNG, GIF, WebP, BMP, TIFF
- **URL requirements:** Must be direct image link (not webpage)
- **File size:** Keep uploads under 10MB for best performance

---

## 📞 Need Help?

- Check `app.py` for API endpoints
- Review `config.py` for settings
- See original `README.md` for model details
- Logs are saved in `logs/app.log` (when web server runs)

---

**Enjoy your enhanced AI Image Detector!** 🎉
