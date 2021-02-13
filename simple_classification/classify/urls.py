from django.urls import path
from . import views
from django.conf.urls import url 

urlpatterns = [
    path('', views.index, name='index'),
    url('predictImage', views.predictImage, name='predictImage')
]
