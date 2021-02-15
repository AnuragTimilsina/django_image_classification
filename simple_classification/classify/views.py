from django.shortcuts import redirect, render
import os
from django.core.files.storage import FileSystemStorage 
from pathlib import Path

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = os.path.join(BASE_DIR, 'models/simple_image_classification.h5')

model = load_model(model_path)


def index(request):
    context = {'test': 1}
    return render(request, 'index.html', context)


def predict(model, image):
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class


def predictImage(request):
    f = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(f.name, f)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    original = load_img(testimage, target_size=(32, 32))
    numpy_image = img_to_array(original)

    prediction = predict(model, numpy_image)

    context = {'filePathName': filePathName, 'prediction': "The Image Says: "+prediction}
    return render(request, 'index.html', context)

class ImagesAndPredcetion:
    def __init__(self,path,prediction):
        self.path=path
        self.prediction=prediction

def viewDataBase(request):
    listOfImages = os.listdir('./media/')
    listOfImagesPath = ['./media/'+i for i in listOfImages]
    listofprediction=list()
    for image_path in listOfImagesPath:
        original = load_img(image_path, target_size=(32, 32))
        numpy_image = img_to_array(original)
        predicted_letter=predict(model,numpy_image)
        listofprediction.append(ImagesAndPredcetion(image_path,predicted_letter))

    context = {'listOfImagesPath': listofprediction}
    return render(request, 'viewDB.html', context)