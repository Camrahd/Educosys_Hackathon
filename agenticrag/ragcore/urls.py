from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("rags/", views.rags_list, name="rags_list"),
    path("rags/check-name/", views.check_rag_name, name="check_rag_name"),
    path("rags/<slug:slug>/", views.rag_detail, name="rag_detail"),
    path("rags/<slug:slug>/ask/", views.ask_question, name="ask_question"),
    # path("upload/", views.upload_files, name="upload_files"),
]
