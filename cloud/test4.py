import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical
import rembg
from PIL import Image
import joblib
from ray.util.joblib import register_ray

# Function to build CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5)
    return model

# Function to predict using the trained model
def predict_digit(model, img_resized):
    pred = model.predict(img_resized)
    predicted_digit = np.argmax(pred)
    return predicted_digit

#--------------TRAINING DATA-------------------

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Reshape and preprocess MNIST dataset
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)

#--------------PREDICT AN IMAGE-------------------

# Load sample image
file = '/home/ble16/cloud/img-num/images[4].jpeg'
test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# Format Image
img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
img_resized = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255

# Register Ray for parallel processing
register_ray()

# Split the training data
split_indices = np.array_split(range(len(x_train)), 2)  # Split data into 4 parts

# Parallelize model training
with joblib.parallel_backend('ray'):
    models = [build_model() for _ in range(2)]  # Create 4 models
    trained_models = [train_model(models[i], x_train[idx], y_train[idx]) for i, idx in enumerate(split_indices)]

# Combine trained models (optional)
# You can choose to combine the trained models or use them individually for predictions.

# Predict using the trained models
predicted_digits = [predict_digit(model, img_resized) for model in trained_models]
print("\nPredictions:", predicted_digits)
