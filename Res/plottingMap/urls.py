from django.urls import path
from . import views


urlpatterns = [
    path('map/', views.plotting_points),
    path('sentiment/', views.sentiment_graph),
    path('cluster/', views.cluster_map),
    path('geojson/', views.geo_json),
    path('interest/', views.inter),

]
