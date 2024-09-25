import os
import cv2
import mtcnn
import dlib
import numpy as np
import tensorflow as tf
import pygame

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize MTCNN detector and facial landmark predictor
mtcnn_detector = mtcnn.MTCNN()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load sound file
pygame.mixer.init()
sound = pygame.mixer.Sound("alarm.wav")

# Function to detect eyes and calculate aspect ratio
def detect_eyes(gray_frame, faces):
    try:
        eyes = []
        for face in faces:
            (x, y, w, h) = face['box']
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eye_rects = eye_cascade.detectMultiScale(roi_gray)
            if len(eye_rects) >= 2:
                eye_rects = sorted(eye_rects, key=lambda x: x[2] * x[3], reverse=True)
                eyes.append(eye_rects[0])
                eyes.append(eye_rects[1])
        return eyes
    except Exception as e:
        print(f"Error in eye detection: {e}")
        return []

# Function to calculate eye aspect ratio
def calculate_ear(eye_points):
    try:
        p1, p2 = eye_points[0], eye_points[1]
        p3, p4 = eye_points[2], eye_points[3]
        p5, p6 = eye_points[4], eye_points[5]

        horizontal_distance = np.sqrt((p6[0] - p2[0])**2 + (p6[1] - p2[1])**2)
        vertical_distance = np.sqrt((p4[0] - p3[0])**2 + (p4[1] - p3[1])**2)

        ear = vertical_distance / horizontal_distance
        return ear
    except Exception as e:
        print(f"Error in calculating EAR: {e}")
        return 0

# Function to find pupil center
def find_pupil_center(eye_image):
    try:
        gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                return (cx, cy)
        return None
    except Exception as e:
        print(f"Error in finding pupil center: {e}")
        return None

# Function to calculate head pose
def calculate_head_pose(landmarks):
    try:
        # Placeholder: Implement head pose estimation
        return None
    except Exception as e:
        print(f"Error in calculating head pose: {e}")
        return None

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

# Function to preprocess image for CNN model
def preprocess_image(image):
    try:
        image_resized = cv2.resize(image, (64, 64))
        image_array = np.array(image_resized).astype('float32') / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Load and preprocess dataset (dummy implementation)
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

# Main loop
cap = cv2.VideoCapture(0)

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

model.fit(train_images, train_labels, epochs=10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(frame)
    if not faces:
        print("No faces detected.")
        continue

    # Detect eyes and calculate EAR
    eyes = detect_eyes(gray, faces)
    ear = 0
    if len(eyes) == 2:
        ear = (calculate_ear(eyes[0]) + calculate_ear(eyes[1])) / 2

    # Detect pupils
    pupil_centers = [find_pupil_center(eye) for eye in eyes]

    # Detect head pose using dlib predictor
    for face in faces:
        (x, y, w, h) = face['box']
        face_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = dlib_predictor(gray, face_rect)
        head_pose = calculate_head_pose(landmarks)

    # Predict drowsiness using the trained CNN model
    preprocessed_frame = preprocess_image(frame)
    if preprocessed_frame is None:
        continue
    drowsiness_prediction = model.predict(preprocessed_frame)

    # Drowsiness detection
    if drowsiness_prediction > 0.5 or ear < 0.2:
        # Raise an alarm
        sound.play()
        cv2.putText(frame, "DROWSINESS DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
