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
from keras_facenet import FaceNet


# Custom accuracy function
def triplet_accuracy(y_true, y_pred):
    # Split the concatenated embeddings back into their components
    # The output shape is likely [batch_size, embedding_dim*3]
    embedding_dim = 512  # FaceNet typically has 512-dimensional embeddings

    # Slice the correct portions from the output tensor
    anchor = y_pred[:, :embedding_dim]
    positive = y_pred[:, embedding_dim:embedding_dim * 2]
    negative = y_pred[:, embedding_dim * 2:]

    # Normalize the embeddings (this improves accuracy)
    anchor = tf.nn.l2_normalize(anchor, axis=1)
    positive = tf.nn.l2_normalize(positive, axis=1)
    negative = tf.nn.l2_normalize(negative, axis=1)

    # Compute cosine similarity instead of Euclidean distance
    # Higher value means more similar
    pos_similarity = tf.reduce_sum(anchor * positive, axis=1)
    neg_similarity = tf.reduce_sum(anchor * negative, axis=1)

    # For accurate face verification, positive similarity should be higher
    correct = tf.cast(pos_similarity > neg_similarity, tf.float32)

    # Return the accuracy as the mean of correct matches
    return tf.reduce_mean(correct)

def triplet_loss(margin=0.5):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Compute Euclidean distance between anchor, positive, and negative
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # Triplet loss formula
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)

    return loss

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

img = cv2.resize(img, (160, 160))     # Resize to match model input

IMG_SHAPE = (160, 160, 3) # ideal for mobileNet


embedding_model= FaceNet()

embedding_model.trainable = False

anchor_input = layers.Input(shape=(160, 160, 3), name='anchor')
positive_input = layers.Input(shape=(160, 160, 3), name='positive')
negative_input = layers.Input(shape=(160, 160, 3), name='negative')

class NormalizeLayer(layers.Layer):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def call(self, inputs):
        # Normalize the input tensor
        return (tf.cast(inputs, tf.float32) - 127.5) / 128.0

normalize_layer = NormalizeLayer()

anchor_input = normalize_layer(anchor_input)
positive_input = normalize_layer(positive_input)
negative_input = normalize_layer(negative_input)

# Get embeddings for anchor, positive, and negative inputs
anchor_embedding = embedding_model.model(anchor_input)
positive_embedding = embedding_model.model(positive_input)
negative_embedding = embedding_model.model(negative_input)

output = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)

model = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=output)
model.compile(optimizer='adam',loss=triplet_loss(), metrics=[triplet_accuracy])
model.summary()

lfw,info= tfds.load("lfw", split="train", with_info=True, as_supervised=True)
dataset = tf.keras.utils.image_dataset_from_directory(
    "lfw/",
    image_size=(160, 160),
    label_mode='int'  # for triplets, we just need the label to pick anchors
)


class TripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, image_size=(160, 160), shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.class_names = [cls for cls in os.listdir(directory)
                            if len(os.listdir(os.path.join(directory, cls))) > 1]  # Only classes with more than 1 image
        self.indexes = np.arange(len(self.class_names))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.class_names) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        anchors = []
        positives = []
        negatives = []

        for i in batch_indexes:
            anchor_class = self.class_names[i]
            anchor_path = os.path.join(self.directory, anchor_class)
            anchor_image = self.load_random_image(anchor_path)
            positive_image = self.load_random_image(anchor_path)
            negative_class = np.random.choice([cls for cls in self.class_names if cls != anchor_class])
            negative_path = os.path.join(self.directory, negative_class)
            negative_image = self.load_random_image(negative_path)

            anchors.append(anchor_image)
            positives.append(positive_image)
            negatives.append(negative_image)

        # Convert to tensors and return as tuple
        anchors_tensor = tf.convert_to_tensor(np.array(anchors), dtype=tf.float32)
        positives_tensor = tf.convert_to_tensor(np.array(positives), dtype=tf.float32)
        negatives_tensor = tf.convert_to_tensor(np.array(negatives), dtype=tf.float32)

        return (anchors_tensor, positives_tensor, negatives_tensor), np.zeros((self.batch_size, 1))

    def load_random_image(self, class_path):
        # Get a random image from the class directory
        img_files = os.listdir(class_path)
        img_file = np.random.choice(img_files)
        img_path = os.path.join(class_path, img_file)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = img[..., ::-1]  # Convert RGB to BGR (critical step)
        img = tf.image.resize(img, self.image_size)
        return img



data_generator = TripletGenerator(directory="lfw", batch_size=32, image_size=(160, 160))

loss,accuracy=model.evaluate(data_generator)

print ('accuarcy:',accuracy)
