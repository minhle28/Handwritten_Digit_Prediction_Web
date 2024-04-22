from django.shortcuts import render
from django.http import HttpResponse
# digit_recognition_app/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
import cv2
import json
import numpy as np
import base64
import matplotlib.pyplot as plt
import time

def home(request):
    return render(request, 'home.html')

# Load the saved model
saved_model_path = '/home/ble16/python_frontend/digit_recognition/digit_recognition/final_model_epoch200.h5'
model = load_model(saved_model_path)


@csrf_exempt
def digit_recognition(request):
    if request.method == 'POST':
        start_time = time.time()  # Record the start time
        data = request.body.decode('utf-8')
        data = json.loads(data)
        image_data = data['image']

        #print("Received image data:", image_data)

        if image_data:
            img_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.bitwise_not(img_resized)

            img_resized = img_resized.reshape(1, 28, 28, 1).astype('float32') / 255
            

            predictions = model.predict(img_resized)[0]
            
            predicted_digit = np.argmax(model.predict(img_resized))
            print('Prediction:', predicted_digit)

            # Calculate percentages for all digits
            percentages = [(round(prob * 100, 2)) for prob in predictions]
            print(percentages)

            # Calculate the latency
            end_time = time.time()
            #latency = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds and round to 2 decimal places
            latency = round(end_time - start_time, 2)  # Latency in seconds, rounded to 2 decimal places

            return JsonResponse({'predicted_digit': int(predicted_digit), 'percentages': percentages, 'latency': latency})
        else:
            return JsonResponse({'error': 'No image data received.'})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})
