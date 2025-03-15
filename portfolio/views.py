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
import base64
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
# from tensorflow.keras.models import load_model

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

# @csrf_exempt
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
# def object_detection(request):
#     """Send image to FastAPI and return the processed image."""
#     if request.method == 'POST' and request.FILES.get('image'):
#         image_file = request.FILES['image']

#         try:
#             # Resize the image before sending
#             img = Image.open(image_file)
#             img = img.convert("RGB")
#             img.thumbnail((640, 640))  # Resize while maintaining aspect ratio
            
#             # Convert to bytes
#             img_io = io.BytesIO()
#             img.save(img_io, format='JPEG')
#             img_io.seek(0)

#             # Send to FastAPI
#             response = requests.post(FASTAPI_URL, files={'file': img_io}, timeout=300)

#             if response.status_code == 200:
#                 return HttpResponse(response.content, content_type="image/jpeg")
#             else:
#                 return JsonResponse({'error': f'FastAPI error: {response.text}'}, status=response.status_code)

#         except requests.exceptions.RequestException as e:
#             return JsonResponse({'error': f'Request failed: {str(e)}'}, status=500)

#     return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def object_detection(request):
    """Send image to FastAPI and return the processed image."""
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        try:
            # Resize image before sending
            img = Image.open(image_file).convert("RGB")
            img.thumbnail((640, 640))  # Resize while maintaining aspect ratio

            # Convert to bytes
            img_io = io.BytesIO()
            img.save(img_io, format='JPEG')
            img_io.seek(0)

            # Send image to FastAPI
            files = {'file': ('image.jpg', img_io, 'image/jpeg')}
            response = requests.post(FASTAPI_URL, files=files, timeout=30)

            if response.status_code == 200:
                return HttpResponse(response.content, content_type=response.headers.get("content-type", "image/jpeg"))
            else:
                return JsonResponse({'error': f'FastAPI error: {response.text}'}, status=response.status_code)

        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': f'Request failed: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)
def object_detection_live_yolo_v8(request):

    return render(request, "live_object_detection_yolo_v8.html")  

@csrf_exempt
def object_detectin_yolo_v8(request):
    print("Received request:", request.method)  # Debugging log

    if request.method == "POST":
        if "image" not in request.FILES:
            print("No image file found in request.")  # Debugging log
            return JsonResponse({"error": "No image file found in request"}, status=400)

        image_file = request.FILES["image"]
        print("Image received:", image_file.name, image_file.size)  # Debugging log

        files = {"file": image_file}
        
        try:
            # Send the image to FastAPI for prediction
            response = requests.post("http://127.0.0.1:8100/live-object-detection/", files=files)
            # print("FastAPI Response:", response.text)  # Debugging log

            return JsonResponse(response.json())

        except requests.exceptions.RequestException as e:
            print("Request failed:", str(e))  # Debugging log
            return JsonResponse({"error": "Failed to connect to FastAPI"}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)



from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from portfolio.detector import run_inference, visualize
import tensorflow as tf

def image_classification(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Save uploaded image temporarily in memory
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Run object detection
        boxes, scores, classes = run_inference(temp_path)

        # Visualize results
        detected_image = visualize(temp_path, boxes, scores, classes)

        # Convert processed image to base64 for frontend display
        _, buffer = cv2.imencode(".jpg", detected_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return JsonResponse({"image": image_base64})

    return render(request, "Image_Classification.html")



FASTAPI_URL2 = "http://127.0.0.1:8001/analyze-face/"

@csrf_exempt
def live_face_analysis(request):
    if request.method == "POST":
        # Read image from frontend
        image_data = request.FILES.get("image")
        if not image_data:
            return JsonResponse({"error": "No image received"})

        # Convert image to OpenCV format
        nparr = np.frombuffer(image_data.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Send frame to FastAPI
        _, buffer = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        response = requests.post(FASTAPI_URL2, files=files)

        if response.status_code == 200:
            result = response.json()

            # Draw bounding boxes and labels
            for face in result.get("faces", []):
                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                age, gender, emotion = face["age"], face["gender"], face["emotion"]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{gender}, {age}, {emotion}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert processed frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            return JsonResponse({"image": image_base64, "faces": result["faces"]})

        return JsonResponse({"error": "Face analysis failed"})

    return JsonResponse({"error": "Invalid request"})


def live_face_analysis_view(request):
    return render(request, "live_face_analysis.html")


def Math_solver(request):
    return render(request, "Math_solver.html")

FASTAPI_URL_MathSolver = "http://127.0.0.1:8080/solve-math/"  # FastAPI endpoint
@csrf_exempt
def solve_math(request):
    if request.method == "POST":
        user_text = request.POST.get("user_text", "")
        image = request.FILES.get("image", None)

        # Prepare the data for FastAPI
        files = {"image": (image.name, image, image.content_type)} if image else None
        data = {"user_text": user_text}

        try:
            response = requests.post(FASTAPI_URL_MathSolver, data=data, files=files)
            
            if response.status_code == 200:
                response_data = response.json()
            else:
                response_data = {"error": f"FastAPI Error: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            response_data = {"error": f"Failed to connect: {str(e)}"}

        return JsonResponse(response_data)

    return JsonResponse({"error": "Invalid request method."})