"""
Enhanced AI Image Detector
Supports both local files and internet URLs
"""
import os
import cv2
import numpy as np
from keras import models
import requests
from PIL import Image
from io import BytesIO
import argparse
import sys


class AIImageDetector:
    def __init__(self, model_path='image_classifier_model.keras'):
        """Initialize the detector with a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = models.load_model(model_path)
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, img_array):
        """
        Common preprocessing for all images
        Handles different color formats and normalizes
        """
        # Handle different color formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:
            # Already BGR, but might be RGB from PIL
            pass
        
        # Resize to model's expected input size
        img_array = cv2.resize(img_array, (128, 128))
        
        # Normalize pixel values to 0-1 range
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects batches)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_from_path(self, image_path):
        """Predict from local file path"""
        if not os.path.exists(image_path):
            return None, None, f"❌ File not found: {image_path}"
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None, "❌ Unable to read image file"
            
            return self._predict(img)
        except Exception as e:
            return None, None, f"❌ Error reading file: {str(e)}"
    
    def predict_from_url(self, image_url):
        """Predict from internet URL"""
        try:
            print(f"📥 Downloading image from: {image_url}")
            
            # Set timeout and headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, timeout=15, headers=headers)
            response.raise_for_status()
            
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return None, None, f"❌ URL does not point to an image (Content-Type: {content_type})"
            
            # Open image with PIL
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            
            # Convert RGB to BGR for OpenCV compatibility
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            print("✅ Image downloaded successfully!")
            return self._predict(img_array)
            
        except requests.exceptions.Timeout:
            return None, None, "❌ Request timed out (15 seconds limit)"
        except requests.exceptions.ConnectionError:
            return None, None, "❌ Connection error - check your internet or URL"
        except requests.exceptions.HTTPError as e:
            return None, None, f"❌ HTTP Error: {e}"
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"
    
    def _predict(self, img_array):
        """Internal prediction method"""
        try:
            # Preprocess the image
            preprocessed = self.preprocess_image(img_array)
            
            # Make prediction (suppress verbose output)
            prediction = self.model.predict(preprocessed, verbose=0)
            
            # Get class with highest probability
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            # Map index to class name
            class_name = 'AI_generated' if class_idx == 0 else 'Natural'
            
            return class_name, confidence, None
            
        except Exception as e:
            return None, None, f"❌ Prediction error: {str(e)}"
    
    def batch_predict(self, sources):
        """Predict multiple images at once"""
        results = []
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}/{len(sources)}] Processing: {source}")
            
            if source.startswith(('http://', 'https://')):
                result = self.predict_from_url(source)
            else:
                result = self.predict_from_path(source)
            
            results.append((source, *result))
        
        return results


def print_result(source, class_name, confidence, error):
    """Pretty print prediction result"""
    print("\n" + "="*60)
    if error:
        print(error)
    else:
        emoji = '🤖' if class_name == 'AI_generated' else '📸'
        print(f"{emoji} Source: {source}")
        print(f"   Prediction: {class_name}")
        print(f"   Confidence: {confidence:.2%}")
    print("="*60)


def interactive_mode():
    """Run in interactive mode for continuous predictions"""
    detector = AIImageDetector()
    
    print("\n" + "="*60)
    print("🔍 AI Image Detector - Interactive Mode")
    print("="*60)
    print("\nSupported inputs:")
    print("  • Local file path: duck.jpg")
    print("  • Internet URL: https://example.com/image.jpg")
    print("  • Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        try:
            source = input("\n📸 Enter image path or URL: ").strip()
            
            if source.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using AI Image Detector!")
                break
            
            if not source:
                print("⚠️  Please enter a valid path or URL")
                continue
            
            # Determine if URL or local path
            if source.startswith(('http://', 'https://')):
                class_name, conf, error = detector.predict_from_url(source)
            else:
                class_name, conf, error = detector.predict_from_path(source)
            
            # Print result
            print_result(source, class_name, conf, error)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='AI Image Detector - Classify images as AI-generated or Natural',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_detector.py duck.jpg
  python enhanced_detector.py https://example.com/image.jpg
  python enhanced_detector.py --batch image1.jpg https://url.com/img.jpg image2.png
  python enhanced_detector.py  (interactive mode)
        """
    )
    
    parser.add_argument(
        'source',
        nargs='?',
        help='Image path or URL (omit for interactive mode)'
    )
    parser.add_argument(
        '--batch',
        nargs='+',
        help='Process multiple images'
    )
    parser.add_argument(
        '--model',
        default='image_classifier_model.keras',
        help='Path to model file (default: image_classifier_model.keras)'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        detector = AIImageDetector(args.model)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Make sure the model file exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Batch mode
    if args.batch:
        results = detector.batch_predict(args.batch)
        
        print("\n" + "="*60)
        print("📊 BATCH RESULTS SUMMARY")
        print("="*60)
        
        for source, class_name, conf, error in results:
            print_result(source, class_name, conf, error)
    
    # Single prediction mode
    elif args.source:
        if args.source.startswith(('http://', 'https://')):
            class_name, conf, error = detector.predict_from_url(args.source)
        else:
            class_name, conf, error = detector.predict_from_path(args.source)
        
        print_result(args.source, class_name, conf, error)
    
    # Interactive mode (no arguments)
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
