from keras.models import load_model
import cv2
import numpy as np

# Load the saved model
saved_model_path = '/home/ble16/cloud/img-num/final_model.h5'
model = load_model(saved_model_path)

# Load sample image
file = '/home/ble16/cloud/img-num/images[9].jpeg'

test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# Format Image
img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)

# Reshape and preprocess the image
img_resized = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255

# Predict using the loaded model
predicted_digit = np.argmax(model.predict(img_resized))

print("\nPrediction:", predicted_digit)
