import os
import cv2
import numpy as np
from keras import models

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image at '{image_path}' not found.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read image.")
        return None
    img = cv2.resize(img, (128, 128))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img


model = models.load_model('image_classifier_model.keras')

def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None, None
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)  
    confidence = prediction[0][class_idx]  
    class_name = 'AI_generated' if class_idx == 0 else 'Natural'
    return class_name, confidence


image_path = "aiimag.jpg"  
class_name, confidence = predict_image(image_path)

if class_name is not None:
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.4f}")
