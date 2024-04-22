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

#--------------TRAINING DATA-------------------

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

# Reshape and preprocess MNIST dataset
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)



#--------------FORMAT AN IMAGE TO MATCH THE MNIST DATASET-------------------

# Load sample image
file = '/home/ble16/cloud/img-num/image[2].png'
test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# Preview an image
plt.imshow(test_image, cmap='gray')
plt.savefig('/home/ble16/cloud/img-num/Preview_Img.png')
print('\nSaved Preview Image.\n')


# Format Image
img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)

# Preview reformatted imagea
plt.imshow(img_resized, cmap='gray')

# Reshape and preprocess the image
img_resized = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255
plt.savefig('/home/ble16/cloud/img-num/Formatted_Img.png')
print('Saved Formatted Image Successfully.\n')



#--------------PREDICT AN IMAGE-------------------

# Predict using the trained model
pred = model.predict(img_resized)
predicted_digit = np.argmax(pred)

print("\nPrediction:", predicted_digit)
plt.savefig('/home/ble16/cloud/img-num/predicted_digit.png')
