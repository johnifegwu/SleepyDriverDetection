import os
import cv2
import mtcnn
import dlib
import numpy as np
import tensorflow as tf
import pygame

# Set a threshold for model accuracy (efficiency score)
ACCURACY_THRESHOLD = 0.85  # Example: 85% accuracy threshold

# Main loop
cap = cv2.VideoCapture(0)

# Function to create CNN model
def create_cnn_model():
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error in creating CNN model: {e}")
        return None

# Load and preprocess dataset
def load_dataset(path):
    try:
        image_size = (64, 64)  # Define the image size (same as expected by CNN)
        images = []
        labels = []
        
        # Loop through each folder in the dataset directory
        for label_folder in os.listdir(path):
            label_folder_path = os.path.join(path, label_folder)
            
            if not os.path.isdir(label_folder_path):
                continue  # Skip if not a directory
            
            # Assign a numeric label based on folder name
            # Assuming folder names correspond to class labels (e.g., "0", "1" for binary classification)
            label = int(label_folder)
            
            # Loop through each image in the folder
            for image_name in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_name)
                
                # Load the image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to load image: {image_path}")
                    continue
                
                # Resize the image to the expected size
                image_resized = cv2.resize(image, image_size)
                
                # Normalize the image by scaling pixel values to [0, 1]
                image_resized = image_resized.astype('float32') / 255.0
                
                # Append image and label to respective lists
                images.append(image_resized)
                labels.append(label)
        
        # Convert lists to NumPy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    except Exception as e:
        print(f"Error in loading dataset: {e}")
        return None, None


# Load and preprocess dataset
train_images, train_labels = load_dataset("models/ddd_dataset")
if train_images is None or train_labels is None:
    print("Failed to load dataset. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Train the model
model = create_cnn_model()
if model is None:
    print("Failed to create CNN model. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Split dataset into training and validation sets
split_index = int(0.8 * len(train_images))  # Use 80% for training, 20% for validation
x_train, x_val = train_images[:split_index], train_images[split_index:]
y_train, y_val = train_labels[:split_index], train_labels[split_index:]

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(x_val, y_val)

# Check if the accuracy is above the threshold and save the model
if accuracy >= ACCURACY_THRESHOLD:
    model.save("trained_models/sleepy_driver_detection_model.h5")
    print(f"Model saved with accuracy: {accuracy * 100:.2f}%")
else:
    print(f"Model not saved. Accuracy ({accuracy * 100:.2f}%) did not meet the threshold of {ACCURACY_THRESHOLD * 100:.2f}%")
