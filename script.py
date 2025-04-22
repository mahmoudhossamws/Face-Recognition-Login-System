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
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    # Normalize for FaceNet
    return (bgr.astype(np.float32) - 127.5) / 128.0

class TripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, image_size=(160, 160), shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.directory = os.path.abspath(directory)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        # Filter valid classes with at least 2 images
        self.class_names = []
        for cls in os.listdir(directory):
            cls_path = os.path.join(directory, cls)
            if os.path.isdir(cls_path):
                valid_images = [f for f in os.listdir(cls_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(valid_images) >= 2:
                    self.class_names.append(cls)

        self.indexes = np.arange(len(self.class_names))
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.class_names) / self.batch_size))

    def __getitem__(self, index):
        anchors, positives, negatives = [], [], []

        # Get batch classes
        batch_classes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        for class_idx in batch_classes:
            while True:  # Keep trying until we get valid data
                anchor_class = self.class_names[class_idx]
                anchor_path = os.path.join(self.directory, anchor_class)

                # Load anchor/positive with validation
                anchor, positive = self.load_pair(anchor_path)
                if anchor is None or positive is None:
                    continue  # Skip invalid pairs

                # Load negative with validation
                negative_class = np.random.choice([c for c in self.class_names if c != anchor_class])
                negative_path = os.path.join(self.directory, negative_class)
                negative = self.load_single(negative_path)
                if negative is None:
                    continue  # Skip invalid negatives

                # Add valid triplet to batch
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
                break  # Exit loop once valid triplet found

        return (tf.convert_to_tensor(anchors),
                tf.convert_to_tensor(positives),
                tf.convert_to_tensor(negatives)), tf.zeros(len(anchors))

    def load_pair(self, class_path):
        """Load two different images from same class"""
        images = []
        while len(images) < 2:
            img = self.load_single(class_path)
            if img is not None:
                images.append(img)
        return images[0], images[1]

    def load_single(self, class_path):
        """Load and preprocess single image"""
        try:
            files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files:
                return None
            img_path = os.path.join(class_path, np.random.choice(files))

            # Read and preprocess
            img = cv2.imread(img_path)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return preprocess_image(img_rgb)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None


lfw,info= tfds.load("lfw", split="train", with_info=True, as_supervised=True)
dataset = tf.keras.utils.image_dataset_from_directory(
    "lfw/",
    image_size=(160, 160),
    label_mode='int'  # for triplets, we just need the label to pick anchors
)

data_generator = TripletGenerator(directory="lfw", batch_size=32, image_size=(160, 160))

loss,accuracy=model.evaluate(data_generator)

print ('accuarcy:',accuracy)
