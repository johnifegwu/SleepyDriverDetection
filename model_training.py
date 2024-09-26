import os
import cv2
import numpy as np
import tensorflow as tf

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

# Load and preprocess dataset (updated with label mapping)
def load_dataset(path):
    try:
        image_size = (64, 64)  # Define the image size (same as expected by CNN)
        images = []
        labels = []

        # Map string labels to integers
        label_mapping = {'Closed': 0, 'Opened': 1}  # Modify as needed
        
        # Loop through each folder in the dataset directory
        for label_folder in os.listdir(path):
            label_folder_path = os.path.join(path, label_folder)
            
            if not os.path.isdir(label_folder_path):
                continue  # Skip if not a directory
            
            # Map the folder name (label) to an integer using the label_mapping dictionary
            if label_folder in label_mapping:
                label = label_mapping[label_folder]
            else:
                print(f"Warning: Label '{label_folder}' not in label_mapping. Skipping...")
                continue  # Skip if the label is not in the mapping
            
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


# Load and preprocess training dataset
train_images, train_labels = load_dataset("models/ddd_dataset/TrainingSet")
if train_images is None or train_labels is None:
    print("Failed to load training dataset. Exiting...")
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

# Train the model with the entire training set (no splitting)
model.fit(train_images, train_labels, epochs=10)

# Load the improvement dataset (for model improvement)
improve_images, improve_labels = load_dataset("models/ddd_dataset/ImprovementSet")
if improve_images is not None and improve_labels is not None:
    print("Performing additional training with the ImprovementSet...")
    model.fit(improve_images, improve_labels, epochs=5)  # Use fewer epochs for improvement phase

# Load the test dataset (for final evaluation)
test_images, test_labels = load_dataset("models/ddd_dataset/testSet")
if test_images is not None and test_labels is not None:
    print("Evaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Save the model only if the test accuracy meets the threshold
    if test_accuracy >= ACCURACY_THRESHOLD:
        model.save("trained_models/sleepy_driver_detection_model.h5")
        print(f"Model saved with test accuracy: {test_accuracy * 100:.2f}%")
    else:
        print(f"Model not saved. Test accuracy ({test_accuracy * 100:.2f}%) did not meet the threshold of {ACCURACY_THRESHOLD * 100:.2f}%")
else:
    print("Test dataset could not be loaded.")
