from django.urls import path
from . import views
urlpatterns = [
    path('', views.index_view, name='index'),
    path('Web-design', views.Web_design_card_detail, name='Web_design_card_detail'),
    path('Finance', views.Finance_card_detail, name='Finance_card_detail'),
    path('Graphic-design', views.Graphic_design_card_detail, name='Graphic_design_card_detail'),
    path('logo-design', views.Logo_design_card_detail, name='Logo_design_card_detail'),
    path('advertisement', views.Advertisement_card_detail, name='Advertisement_card_detail'),
    path('video-content', views.Video_content_card_detail, name='Video_content_card_detail'),
    path('viral-tweet', views.Viral_tweet_card_detail, name='Viral_tweet_card_detail'),
    path('investment', views.Investment_card_detail, name='Investment_card_detail'),
    path('composing-song', views.Composing_song_card_detail, name='Composing_song_card_detail'),
    path('online-song', views.Online_song_card_detail, name='Online_song_card_detail'),
    path('podcast', views.Podcast_card_detail, name='Podcast_card_detail'),
    path('Graduation', views.Graduation_card_detail, name='Graduation_card_detail'),
    path('Educator', views.Educator_card_detail, name='Educator_card_detail'),
    path('contact', views.contact, name='contact'),
    path('Topics-listing', views.Topics_listing, name='Topics_listing'),
    path("object-detection-view/", views.object_detection_view, name="object_detection_view"),
    path("object-detection/", views.object_detection, name="object_detection"),
    
]