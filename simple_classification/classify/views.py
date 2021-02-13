from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
#import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback
from tensorflow import keras
#from PIL import Image
import os

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
model_path = os.path.join(BASE_DIR, 'simple_image_classification.h5')

model = keras.models.load_model(model_path)


def predict(model, image):
  class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  return predicted_class


def index(request):

    if request.method == "POST":
        f = request.FILES['sentFile']  # here you get the files needed

        response = {}
        file_name = "tb.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        #file_url = default_storage.url(os.path.join(BASE_DIR, file_name_2))
        original = load_img(file_url, target_size=(32, 32))
        numpy_image = img_to_array(original)

        prediction = predict(model, numpy_image)
        response['name'] = str(prediction)
        return render(request, 'home.html', response)

    else:
        return render(request, 'home.html')
