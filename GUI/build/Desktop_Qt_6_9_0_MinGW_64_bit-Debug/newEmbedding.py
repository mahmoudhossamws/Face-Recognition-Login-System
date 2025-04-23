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

def add_to_database(name, embedding, csv_path="users_database.csv"):
    """
    Appends a new face embedding to the database CSV
    Args:
        name (str): Person's name
        embedding (np.array): FaceNet embedding vector (512-dim)
        csv_path (str): Path to CSV database
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Convert embedding to string format
        embedding_str = ' '.join([str(x) for x in embedding])

        # Create new record
        new_record = pd.DataFrame([[name, embedding_str]],
                                  columns=['name', 'embedding'])

        # Check if file exists and is writable
        if os.path.exists(csv_path):
            if not os.access(csv_path, os.W_OK):
                # Try alternative location if original isn't writable
                csv_path = os.path.join(os.path.expanduser("~"), "users_data.csv")

        # Append to existing file or create new
        if os.path.exists(csv_path):
            new_record.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            new_record.to_csv(csv_path, index=False)

        print(f"Successfully added {name} to database at {csv_path}")
        return True

    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python newEmbedding.py <username>")
        sys.exit(1)

    user_name = sys.argv[1]
    photo = cv2.imread('person.jpg')

    preprocessed_photo = preprocess_image(photo)
    if preprocessed_photo is None:
        print("Face not detected!")
        sys.exit(1)

    new_embedding = embedding_model.embeddings(np.expand_dims(preprocessed_photo, axis=0))[0]
    add_to_database(user_name, new_embedding)

