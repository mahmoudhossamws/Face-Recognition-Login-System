import tensorflow as tf
import keras
import os
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import random
import shutil
import pandas as pd
from keras_facenet import FaceNet
import sys


def triplet_accuracy(y_true, y_pred):
    emb_dim = 512
    anchor, pos, neg = y_pred[..., :emb_dim], y_pred[..., emb_dim:2*emb_dim], y_pred[..., 2*emb_dim:]
    pos_sim = tf.reduce_sum(anchor * pos, axis=1)
    neg_sim = tf.reduce_sum(anchor * neg, axis=1)
    return tf.reduce_mean(tf.cast(pos_sim > neg_sim, tf.float32))

def triplet_loss(margin=0.5):
    def loss(_, y_pred):
        anchor, pos, neg = y_pred[..., :512], y_pred[..., 512:1024], y_pred[..., 1024:]
        pos_dist = tf.reduce_sum(tf.square(anchor - pos), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - neg), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0))
    return loss

IMG_SHAPE = (160, 160, 3) # ideal for facenet

# Initialize FaceNet model
embedding_model = FaceNet()
embedding_model.trainable = False

# Build triplet model
anchor_input = layers.Input((160, 160, 3), name='anchor')
positive_input = layers.Input((160, 160, 3), name='positive')
negative_input = layers.Input((160, 160, 3), name='negative')

embeddings = [
    embedding_model.model(anchor_input),
    embedding_model.model(positive_input),
    embedding_model.model(negative_input)
]

output = layers.concatenate(embeddings)
model = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input],
    outputs=output
)

model.compile(optimizer='adam',loss=triplet_loss(), metrics=[triplet_accuracy])

model.summary()



mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def align_face(image):
    """Align and crop face using MediaPipe"""
    results = face_detector.process(image)
    if not results.detections:
        return None

    # Get first face bounding box
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w = image.shape[:2]
    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
    width, height = int(bbox.width * w), int(bbox.height * h)

    # Expand crop by 20% for margin
    y = max(0, y - int(0.2 * height))
    x = max(0, x - int(0.2 * width))
    height = min(h - y, int(1.4 * height))
    width = min(w - x, int(1.4 * width))

    return image[y:y + height, x:x + width]

def preprocess_image(img):
    """Full preprocessing pipeline"""
    # Align face
    aligned = align_face(img)
    if aligned is None:
        return None

    # Resize and convert to BGR
    resized = cv2.resize(aligned, (160, 160))

    return resized

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)


def find_best_match(new_embedding, csv_path, threshold=0.55):
    """Returns either (best_match_name, similarity_score) or None"""
    df = pd.read_csv(csv_path)

    best_match = None
    best_score = -1

    for _, row in df.iterrows():
        stored_embedding = np.fromstring(row['embedding'], sep=' ')
        stored_embedding = normalize_embedding(stored_embedding)

        similarity = np.dot(new_embedding, stored_embedding)
        print(f"Comparing with {row['name']}: similarity={similarity}")

        if similarity > best_score:
            best_score = similarity
            best_match = row['name']

    return (best_match, best_score) if best_score >= threshold else None

photo = cv2.imread('loginAttempt.jpg')
preprocessed_photo = preprocess_image(photo)

if preprocessed_photo is None:
    print("Error: Face not detected or preprocessing failed.")
    sys.exit(1)

new_embedding = embedding_model.embeddings(np.expand_dims(preprocessed_photo, axis=0))[0]
if new_embedding is not None:
    match = find_best_match(new_embedding, "users_database.csv")

if match:
    name,sim = match
    print(f"Best match: {name} sim: {sim}")
else:
    print("No matching face found")
