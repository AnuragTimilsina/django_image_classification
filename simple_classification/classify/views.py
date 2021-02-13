from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage 
from pathlib import Path

from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
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

    response = {}
    f = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(f.name, f)
    filePathName = fs.url(filePathName)
    response['filePathName'] = filePathName
    original = load_img(filePathName, target_size=(32, 32))
    numpy_image = img_to_array(original)

    prediction = predict(model, numpy_image)
    response['name'] = str(prediction)
    return render(request, 'index.html', response)

'''
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
        return render(request, 'index.html', response)

    else:
        return render(request, 'index.html')
'''
