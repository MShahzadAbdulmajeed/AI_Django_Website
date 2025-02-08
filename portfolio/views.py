from django.shortcuts import render
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
# Create your views here.from django.http import HttpResponse
import requests
from django.http import HttpResponse
import io
FASTAPI_URL  = 'http://127.0.0.1:9000/predict/'

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

from django.http import JsonResponse
import cv2
from tensorflow.keras.models import load_model

# def object_detection(request):
#     if request.method == 'POST':
#         # Handle file upload or video frame data
#         image_file = request.FILES.get('image')
#         if not image_file:
#             return JsonResponse({'error': 'No image uploaded'}, status=400)
        
#         # Read the image using PIL
#         img = Image.open(image_file)
#         img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
#         # Get the processed input tensor from preprocess_image function
#         input_tensor, _ = preprocess_image(image_file.read())  # Assuming preprocess_image returns both image and tensor
        
#         # Run inference on your model
#         output_dict = infer(input_tensor)
        
#         # Extract detection data (assuming output_dict has 'boxes', 'scores', etc.)
#         boxes = output_dict['detection_boxes'].numpy()
#         scores = output_dict['detection_scores'].numpy()
#         classes = output_dict['detection_classes'].numpy()

#         # Filter out low confidence predictions
#         filter_threshold = 0.5
#         confident_indices = where(scores > filter_threshold)
        
#         # Draw bounding boxes and labels on the image
#         for i in confident_indices:
#             box = boxes[i]
#             label = category_index[classes[i]]['name']
            
#             # Convert to RGB again for displaying
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             # Here you would need to draw the box and label on the image
            
#         return JsonResponse({'detections': detection_data})
def object_detection_view(request):
    """
    Render the HTML page for live object detection.
    """
    return render(request, "object_detection.html")  # Ensure this file exists in templates/

@csrf_exempt
# def object_detection(request):
#     """
#     Receives image frames from the frontend, sends them to FastAPI for inference,
#     and returns the detected objects along with an image containing bounding boxes.
#     """
#     if request.method == 'POST' and request.FILES.get('image'):
#         image_file = request.FILES['image']

#         # Convert image to bytes
#         img_bytes = BytesIO(image_file.read())

#         # Send image to FastAPI for prediction
#         files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
#         response = requests.post(FASTAPI_URL, files=files)

#         if response.status_code == 200:
#             detection_data = response.json()

#             # Draw bounding boxes on the image
#             img = Image.open(img_bytes)
#             img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#             for obj in detection_data["detections"]:
#                 x_min, y_min, x_max, y_max = obj["box"]
#                 label = f"{obj['class']} ({obj['score']:.2f})"
                
#                 # Draw bounding box
#                 cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                 # Draw label
#                 cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                             0.5, (0, 255, 0), 2, cv2.LINE_AA)

#             # Convert image to bytes for frontend display
#             _, buffer = cv2.imencode(".jpg", img)

#             return JsonResponse({
#                 "detections": detection_data["detections"],
#                 "image": buffer.tobytes().decode('latin1')  # Convert to string for JSON response
#             })

#         return JsonResponse({"error": "FastAPI error", "details": response.text}, status=500)

#     return JsonResponse({"error": "Invalid request"}, status=400)
def object_detection(request):
    """Send image to FastAPI and return the processed image."""
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        try:
            # Resize the image before sending
            img = Image.open(image_file)
            img = img.convert("RGB")
            img.thumbnail((640, 640))  # Resize while maintaining aspect ratio
            
            # Convert to bytes
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG')
            img_io.seek(0)

            # Send to FastAPI
            response = requests.post(FASTAPI_URL, files={'file': img_io}, timeout=300)

            if response.status_code == 200:
                return HttpResponse(response.content, content_type="image/jpeg")
            else:
                return JsonResponse({'error': f'FastAPI error: {response.text}'}, status=response.status_code)

        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': f'Request failed: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)