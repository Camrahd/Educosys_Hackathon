from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_files, name='upload'),
    path('ask/', views.ask_question, name='ask'),
]
