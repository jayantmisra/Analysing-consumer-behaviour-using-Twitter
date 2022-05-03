
from django.urls import path

from . import views

urlpatterns = [
    path('map/', views.plotting),
    path('md/', views.md)

]
