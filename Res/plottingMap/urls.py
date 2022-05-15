from django.urls import path
from . import views


urlpatterns = [
    path('map/', views.plotting_points),
    path('sentiment/', views.sentiment_graph),

]
