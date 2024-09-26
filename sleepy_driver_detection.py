import cv2
import mtcnn
import dlib
import numpy as np
import tensorflow as tf
import pygame

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('files/haarcascade_eye.xml')

# Initialize MTCNN detector and facial landmark predictor
mtcnn_detector = mtcnn.MTCNN()
dlib_predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Load sound file for alarm
pygame.mixer.init()
sound = pygame.mixer.Sound("media/alarm.mp3")

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


# Function to detect eyes and return landmarks using dlib
def detect_eyes(gray_frame, faces):
    eyes = []
    for face in faces:
        (x, y, w, h) = face['box']
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = dlib_predictor(gray_frame, rect)

        # Left eye landmarks (points 36-41 in dlib's 68 point model)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])

        # Right eye landmarks (points 42-47 in dlib's 68 point model)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        eyes.append(left_eye)
        eyes.append(right_eye)
    
    return eyes


# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    # Coordinates for vertical eye landmarks
    p2 = np.linalg.norm(eye[1] - eye[5])
    p3 = np.linalg.norm(eye[2] - eye[4])
    # Coordinates for horizontal eye landmarks
    p1 = np.linalg.norm(eye[0] - eye[3])
    # Eye aspect ratio
    ear = (p2 + p3) / (2.0 * p1)
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
