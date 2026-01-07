import os
import cv2
import numpy as np
from keras import models, layers, utils
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_dir = r'dataset'  # Use raw string to handle backslashes

# Check if the main dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"Error: Folder '{dataset_dir}' does not exist.")
    exit()

# Check for subfolders
categories = ['AI_generated', 'Natural']
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    if not os.path.exists(category_path):
        print(f"Folder '{category_path}' does not exist.")
        exit()

# Load data and labels
data = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_dir, category)
    class_idx = categories.index(category)

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            data.append(img)
            labels.append(class_idx)

data = np.array(data) / 255.0  # Normalize the images
labels = utils.to_categorical(labels, num_classes=len(categories))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('image_classifier_model.keras')
print("Model trained and saved successfully.")
