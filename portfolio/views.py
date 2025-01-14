from django.shortcuts import render

# Create your views here.

def index_view(request):

    return render(request, 'index.html')
def Web_design_card_detail(request):

    return render(request, 'topics-detail.html')
def Finance_card_detail(request):

    return render(request, 'topics-detail.html')
def Graphic_design_card_detail(request):
    return render(request, 'topics-detail.html')
def Logo_design_card_detail(request):
    return render(request, 'topics-detail.html')
def Advertisement_card_detail(request):
    return render(request, 'topics-detail.html')
def Video_content_card_detail(request):
    return render(request, 'topics-detail.html')
def Viral_tweet_card_detail(request):
    return render(request, 'topics-detail.html')
def Investment_card_detail(request):
    return render(request, 'topics-detail.html')
def Composing_song_card_detail(request):
    return render(request, 'topics-detail.html')
def Online_song_card_detail(request):
    return render(request, 'topics-detail.html')
def Podcast_card_detail(request):
    return render(request, 'topics-detail.html')
def Graduation_card_detail(request):
    return render(request, 'topics-detail.html')
def Educator_card_detail(request):
    return render(request, 'topics-detail.html')
def contact(request):
    return render(request, 'contact.html')
def Topics_listing(request):
    return render(request, 'topics-listing.html')

