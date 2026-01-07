# 🖼️ AI Image Classifier: AI-Generated vs. Natural Images 🖼️

Welcome to the *AI Image Classifier*, a deep learning-based solution designed to distinguish between AI-generated and natural images with high accuracy. Built using Keras and OpenCV, this project leverages a convolutional neural network (CNN) to classify images, making it a perfect tool for exploring the boundaries between human-created and machine-generated visual content.

## 🔍 Project Overview

The *AI Image Classifier* addresses the challenge of identifying whether an image is AI-generated or natural—a task increasingly relevant in the era of advanced generative AI models. Using a CNN trained on a dataset of labeled images, this project achieves reliable classification by learning visual patterns unique to each category. It includes scripts for training the model and testing it on new images, with support for easy integration into larger applications.

### ✨ Key Features:

- *Binary Classification:* Classifies images as either `AI_generated` or `Natural`.
- *CNN Architecture:* Utilizes convolutional layers for robust feature extraction.
- *Image Preprocessing:* Handles resizing and normalization for consistent input.
- *Confidence Scoring:* Provides prediction confidence alongside class labels.
- *Modular Design:* Separate scripts for training (`train_model.py`) and testing (`test_model.py`).

## 🚀 Getting Started

### 1. *Prerequisites:*
- Python 3.x installed on your system.
- Required libraries: Keras, TensorFlow, OpenCV, NumPy, and scikit-learn.
- A dataset with two subfolders: `AI_generated` and `Natural`, each containing respective images.

### 2. *Setting Up:*

- Clone the repository (if hosted on GitHub):
  ```bash
  git clone https://github.com/your-username/AI_Image_Classifier.git
  cd AI_Image_Classifier
  ```

- Install dependencies:
  ```bash
  pip install tensorflow keras opencv-python numpy scikit-learn
  ```

- Prepare your dataset:
  - Create a `dataset/` folder in the project directory.
  - Inside `dataset/`, create two subfolders: `AI_generated` and `Natural`.
  - Place AI-generated images in `dataset/AI_generated/` and natural images in `dataset/Natural/`.

  Example structure:
  ```
  dataset/
  ├── AI_generated/
  │   ├── ai_image1.jpg
  │   ├── ai_image2.jpg
  │   └── ...
  └── Natural/
      ├── natural_image1.jpg
      ├── natural_image2.jpg
      └── ...
  ```

### 3. *Training the Model:*

- Run the training script to build and save the model:
  ```bash
  python train_model.py
  ```
- This script:
  - Loads images from the `dataset/` folder.
  - Preprocesses them (resizes to 128x128 and normalizes pixel values).
  - Trains a CNN with 10 epochs.
  - Saves the trained model as `image_classifier_model.keras`.

### 4. *Testing the Model:*

- Use the testing script to classify a new image:
  ```bash
  python test_model.py
  ```
- Update the `image_path` variable in `test_model.py` to point to your test image (e.g., `E:\My_Projects\image_classifier\AI_Image_Classifier\duck.jpg`).
- The script will output the predicted class (`AI_generated` or `Natural`) and the confidence score.

  Example output:
  ```
  Predicted class: Natural
  Confidence: 0.9234
  ```

### 5. *Sample Data:*
- The project includes a sample image (`duck.jpg`) for testing. Ensure it’s in the project directory or update the path in `test_model.py`.

## 💾 Directory Structure

```
AI_Image_Classifier/
│
├── image_classifier.h5           # Legacy model file (HDF5 format)
├── image_classifier_model.h5     # Legacy model file (HDF5 format)
├── image_classifier_model.keras  # Trained Keras model (current format)
├── README.md                     # Project documentation
├── train_model.py                # Script to train the CNN model
├── test_model.py                 # Script to test the model on new images
└── duck.jpg                      # Sample image for testing (not listed but referenced)
```

### 📝 Code Explanation

1. **train_model.py**:
   - Loads and preprocesses images from the `dataset/` folder.
   - Builds a CNN with two convolutional layers, max-pooling, and dense layers.
   - Trains the model using the Adam optimizer and categorical cross-entropy loss.
   - Saves the trained model as `image_classifier_model.keras`.

2. **test_model.py**:
   - Loads the trained model (`image_classifier_model.keras`).
   - Preprocesses a single input image (resizes to 128x128, normalizes).
   - Predicts the class (`AI_generated` or `Natural`) and confidence score.

## 🌐 System Configuration

- *Model Input:* Images are resized to 128x128 pixels with 3 color channels (RGB).
- *Normalization:* Pixel values are scaled to the range [0, 1] by dividing by 255.0.
- *Hardware:* Can run on CPU, but a GPU is recommended for faster training.
- *Model Format:* Uses Keras’ native `.keras` format for saving/loading the model.

## 🛠️ How It Works

1. *Training* (`train_model.py`):
   - Loads images from `dataset/AI_generated/` and `dataset/Natural/`.
   - Resizes images to 128x128 and normalizes pixel values.
   - Splits data into 80% training and 20% testing sets.
   - Trains a CNN with 32 and 64 filters in convolutional layers, followed by dense layers.
   - Uses categorical cross-entropy loss for binary classification.

2. *Testing* (`test_model.py`):
   - Loads a single image and preprocesses it.
   - Uses the trained model to predict the class and confidence.
   - Outputs the result in a user-friendly format.

## 🎯 Project Intent

The *AI Image Classifier* aims to provide a simple yet effective tool for distinguishing AI-generated images from natural ones. It’s ideal for researchers, developers, or enthusiasts interested in understanding the visual differences between these image types, with potential applications in content moderation, digital forensics, and AI ethics.

## 🔧 Customization

Enhance the project with these ideas:
- *Add More Classes:* Extend the classifier to handle additional categories (e.g., `Synthetic`, `Edited`).
- *Improve Model Architecture:* Add more convolutional layers or use transfer learning with pre-trained models like VGG16 or ResNet.
- *Batch Testing:* Modify `test_model.py` to process multiple images in a folder.
- *Visualize Results:* Use Matplotlib to display the input image alongside its predicted class and confidence.

## 📌 Links
- **Demo Video:** https://youtu.be/ZenM_MKdBV4
