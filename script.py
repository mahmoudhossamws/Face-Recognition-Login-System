import tensorflow as tf
import keras
import os
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Initialize mediapipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load image from file
img = cv2.imread("person.jpg")  # Replace with your image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize face detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(img_rgb)

    # If a face is detected, crop the first face
    if results.detections:
        detection = results.detections[0]  # Take the first detected face
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        img = img[y:y+h, x:x+w]  # Crop the image to the detected face

# Save or display the cropped face image
cv2.imwrite('cropped_face.jpg', img)  # Save the cropped face
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.resize(img, (128, 128))     # Resize to match model input
img = img / 255.0                     # Normalize pixel values

IMG_SHAPE = (128, 128, 3 ) # ideal for mobileNet

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

base_model.trainable = False

model = models.Sequential ([
base_model,
layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=IMG_SHAPE),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu',padding='same'),

layers.Flatten(),

layers.Dense(64, activation='relu')])
