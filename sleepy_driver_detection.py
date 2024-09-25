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

# Load sound file for alarm
pygame.mixer.init()
sound = pygame.mixer.Sound("alarm.wav")

# Load the trained model
model = tf.keras.models.load_model("trained_models/sleepy_driver_detection_model.h5")

# Function to preprocess image for the CNN model
def preprocess_image(image):
    try:
        image_resized = cv2.resize(image, (64, 64))  # Resize to the model's expected input size
        image_array = np.array(image_resized).astype('float32') / 255.0  # Normalize pixel values
        return np.expand_dims(image_array, axis=0)  # Add a batch dimension
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Function to detect eyes and calculate aspect ratio (EAR)
def detect_eyes(gray_frame, faces):
    eyes = []
    for face in faces:
        (x, y, w, h) = face['box']
        roi_gray = gray_frame[y:y + h, x:x + w]
        eye_rects = eye_cascade.detectMultiScale(roi_gray)
        if len(eye_rects) >= 2:
            eye_rects = sorted(eye_rects, key=lambda x: x[2] * x[3], reverse=True)
            eyes.append(eye_rects[0])
            eyes.append(eye_rects[1])
    return eyes

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye
    horizontal_distance = np.sqrt((p6[0] - p2[0])**2 + (p6[1] - p2[1])**2)
    vertical_distance = np.sqrt((p4[0] - p3[0])**2 + (p4[1] - p3[1])**2)
    ear = vertical_distance / horizontal_distance
    return ear

# Start capturing video
cap = cv2.VideoCapture(0)

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

    # Detect eyes
    eyes = detect_eyes(gray, faces)
    
    # If eyes are detected, calculate the EAR
    ear = 0
    if len(eyes) == 2:
        ear = (calculate_ear(eyes[0]) + calculate_ear(eyes[1])) / 2

    # Preprocess the frame and predict drowsiness using the trained model
    preprocessed_frame = preprocess_image(frame)
    if preprocessed_frame is None:
        continue

    drowsiness_prediction = model.predict(preprocessed_frame)[0][0]  # Get the prediction score (sigmoid output)

    # Drowsiness detection logic based on model prediction and EAR
    if drowsiness_prediction > 0.5 or ear < 0.2:  # 0.5 threshold for the model's sigmoid output
        # Raise an alarm if drowsiness is detected
        sound.play()
        cv2.putText(frame, "DROWSINESS DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame with drowsiness warning if detected
    cv2.imshow('Drowsiness Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
