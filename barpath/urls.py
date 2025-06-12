from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analysis/', views.analysis, name='analysis'),
    path('process_video/', views.process_video, name='process_video'),
    path('video_feed/', views.video_feed, name='video_feed'),
] 