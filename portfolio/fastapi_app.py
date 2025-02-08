# # from fastapi import FastAPI, uploadfile, File
# # import tensorflow as tf
# # import numpy as np
# # import cv2
# # from io import BytesIO

# # app = FastAPI()

# # model = tf.saved_model.load('save_model')
# # infer = model.signatures['serving_default']

# # def preprocess_image(image_bytes):
# #     image_np = np.frombuffer(image_bytes, np.uint8)
# #     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
# #     image = cv2.cvtcolor(image, cv2.COLOR_BGR2RGB)
# #     input_tensor = tf.convert_to_tensor(image)
# #     return input_tensor

# # @app.post('/predict/')
# # async def predict(file: UploadFile = File(...)):
# #     image_bytes = await file.read()
# #     image, input_tensor = preprocess_image(image_bytes)

# #     output_dict = infer(input_tensor)

# #     # Extracting useful info
# #     boxes = output_dict["detection_boxes"].numpy()[0]  # Normalized bounding boxes
# #     scores = output_dict["detection_scores"].numpy()[0]  # Confidence scores
# #     classes = output_dict["detection_classes"].numpy()[0].astype(int)  # Class labels

# #     height, width, _ = image.shape
# #     detections = []

# #     for i in range(len(scores)):
# #         if scores[i] > 0.5:  # Confidence threshold
# #             y_min, x_min, y_max, x_max = boxes[i]
# #             x_min, x_max = int(x_min * width), int(x_max * width)
# #             y_min, y_max = int(y_min * height), int(y_max * height)

# #             detections.append({
# #                 "box": [x_min, y_min, x_max, y_max],
# #                 "score": float(scores[i]),
# #                 "class": int(classes[i])
# #             })

# #     return {"detections": detections}


# # Load the model

# # COCO class labels
# COCO_CLASSES = {
#     1: "Person",
#     2: "Bicycle",
#     3: "Car",
#     4: "Motorcycle",
#     5: "Airplane",
#     6: "Bus",
#     7: "Train",
#     8: "Truck",
#     9: "Boat",
#     10: "Traffic Light",
#     11: "Fire Hydrant",
#     13: "Stop Sign",
#     14: "Parking Meter",
#     15: "Bench",
#     16: "Bird",
#     17: "Cat",
#     18: "Dog",
#     19: "Horse",
#     20: "Sheep",
#     21: "Cow",
#     22: "Elephant",
#     23: "Bear",
#     24: "Zebra",
#     25: "Giraffe",
#     27: "Backpack",
#     28: "Umbrella",
#     31: "Handbag",
#     32: "Tie",
#     33: "Suitcase",
#     34: "Frisbee",
#     35: "Skis",
#     36: "Snowboard",
#     37: "Sports Ball",
#     38: "Kite",
#     39: "Baseball Bat",
#     40: "Baseball Glove",
#     41: "Skateboard",
#     42: "Surfboard",
#     43: "Tennis Racket",
#     44: "Bottle",
#     46: "Wine Glass",
#     47: "Cup",
#     48: "Fork",
#     49: "Knife",
#     50: "Spoon",
#     51: "Bowl",
#     52: "Banana",
#     53: "Apple",
#     54: "Sandwich",
#     55: "Orange",
#     56: "Broccoli",
#     57: "Carrot",
#     58: "Hot Dog",
#     59: "Pizza",
#     60: "Donut",
#     61: "Cake",
#     62: "Chair",
#     63: "Couch",
#     64: "Potted Plant",
#     65: "Bed",
#     67: "Dining Table",
#     70: "Toilet",
#     72: "TV",
#     73: "Laptop",
#     74: "Mouse",
#     75: "Remote",
#     76: "Keyboard",
#     77: "Cell Phone",
#     78: "Microwave",
#     79: "Oven",
#     80: "Toaster",
#     81: "Sink",
#     82: "Refrigerator",
#     84: "Book",
#     85: "Clock",
#     86: "Vase",
#     87: "Scissors",
#     88: "Teddy Bear",
#     89: "Hair Drier",
#     90: "Toothbrush"
# }

# # def preprocess_image(image_bytes):
# #     """Preprocess uploaded image for model inference."""
# #     image_np = np.frombuffer(image_bytes, np.uint8)
# #     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     image_resized = cv2.resize(image, (640, 640))  # Resize to model input size
# #     input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
# #     input_tensor = tf.expand_dims(input_tensor, axis=0)
# #     return image, input_tensor


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import Response
# import tensorflow as tf
# import numpy as np
# import cv2

# app = FastAPI()

# # Load the model
# MODEL_PATH = "save_model"
# model = tf.saved_model.load(MODEL_PATH)
# infer = model.signatures['serving_default']

# def preprocess_image(image_bytes):
#     """Preprocess uploaded image for model inference."""
#     image_np = np.frombuffer(image_bytes, np.uint8)
#     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Resize image to (640, 640) if needed
#     image_resized = cv2.resize(image_rgb, (640, 640))
    
#     # Convert to tensor and add batch dimension
#     input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
#     input_tensor = tf.expand_dims(input_tensor, axis=0)

#     # Check image dimensions and data type
#     if image.shape != (640, 640, 3):
#         print("Error: Image size is incorrect", image.shape)
#     if image.dtype != np.uint8:
#         print("Error: Image data type is incorrect", image.dtype)

#     return image_resized, input_tensor

# @app.post('/predict/')
# async def predict(file: UploadFile = File(...)):
#     """Predict objects in the uploaded image and return processed image with bounding boxes."""
#     try:
#         # Read image and preprocess
#         image_bytes = await file.read()
#         image, input_tensor = preprocess_image(image_bytes)

#         # Perform inference
#         output_dict = infer(input_tensor)

#         # Extract detections
#         boxes = output_dict["detection_boxes"].numpy()[0]  # Bounding boxes (normalized)
#         scores = output_dict["detection_scores"].numpy()[0]  # Confidence scores
#         classes = output_dict["detection_classes"].numpy()[0].astype(int)  # Class labels

#         height, width, _ = image.shape
#         detections = []

#         for i in range(len(scores)):
#             if scores[i] > 0.5:  # Confidence threshold
#                 y_min, x_min, y_max, x_max = boxes[i]
#                 x_min, x_max = int(x_min * width), int(x_max * width)
#                 y_min, y_max = int(y_min * height), int(y_max * height)

#                 detections.append({
#                     "box": [x_min, y_min, x_max, y_max],
#                     "score": float(scores[i]),
#                     "class": int(classes[i])
#                 })

#                 # Draw bounding box
#                 cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                 # Draw label
#                 label = f"Class {classes[i]} ({scores[i]:.2f})"
#                 cv2.putText(image, label, (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Encode image as JPEG
#         _, encoded_image = cv2.imencode(".jpg", image)

#         # Return processed image as response
#         return Response(content=encoded_image.tobytes(), media_type="image/jpeg",
#                         headers={"Content-Disposition": 'inline; filename="result.jpg"'})

#     except Exception as e:
#         return {"error": str(e)}
COCO_CLASSES = {
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorcycle",
    5: "Airplane",
    6: "Bus",
    7: "Train",
    8: "Truck",
    9: "Boat",
    10: "Traffic Light",
    11: "Fire Hydrant",
    13: "Stop Sign",
    14: "Parking Meter",
    15: "Bench",
    16: "Bird",
    17: "Cat",
    18: "Dog",
    19: "Horse",
    20: "Sheep",
    21: "Cow",
    22: "Elephant",
    23: "Bear",
    24: "Zebra",
    25: "Giraffe",
    27: "Backpack",
    28: "Umbrella",
    31: "Handbag",
    32: "Tie",
    33: "Suitcase",
    34: "Frisbee",
    35: "Skis",
    36: "Snowboard",
    37: "Sports Ball",
    38: "Kite",
    39: "Baseball Bat",
    40: "Baseball Glove",
    41: "Skateboard",
    42: "Surfboard",
    43: "Tennis Racket",
    44: "Bottle",
    46: "Wine Glass",
    47: "Cup",
    48: "Fork",
    49: "Knife",
    50: "Spoon",
    51: "Bowl",
    52: "Banana",
    53: "Apple",
    54: "Sandwich",
    55: "Orange",
    56: "Broccoli",
    57: "Carrot",
    58: "Hot Dog",
    59: "Pizza",
    60: "Donut",
    61: "Cake",
    62: "Chair",
    63: "Couch",
    64: "Potted Plant",
    65: "Bed",
    67: "Dining Table",
    70: "Toilet",
    72: "TV",
    73: "Laptop",
    74: "Mouse",
    75: "Remote",
    76: "Keyboard",
    77: "Cell Phone",
    78: "Microwave",
    79: "Oven",
    80: "Toaster",
    81: "Sink",
    82: "Refrigerator",
    84: "Book",
    85: "Clock",
    86: "Vase",
    87: "Scissors",
    88: "Teddy Bear",
    89: "Hair Drier",
    90: "Toothbrush"
}


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# Load the model
MODEL_PATH = "save_model"
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures['serving_default']

# def preprocess_image(image_bytes):
#     """Preprocess uploaded image for model inference."""
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Open and convert to RGB
#     img = np.array(img)  # Convert to NumPy array
#     img = tf.image.resize(img, (640, 640))  # Resize to model input size
#     img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure uint8 format
#     img_tensor = tf.convert_to_tensor(img)  # Convert to Tensor
#     img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

#     return img, img_tensor  # Return both original image and tensor
def preprocess_image(image_bytes):
    """Efficiently preprocess image for model inference."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((640, 640))  # Resize once instead of multiple times
    img = np.array(img, dtype=np.uint8)  # Ensure uint8 format
    img = tf.convert_to_tensor(img)  # Convert to Tensor
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img
@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    """Predict objects in the uploaded image and return processed image with bounding boxes."""
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image, input_tensor = preprocess_image(image_bytes)

        # Perform inference
        output_dict = infer(input_tensor)

        # Extract detections
        boxes = output_dict["detection_boxes"].numpy()[0]  # Normalized bounding boxes
        scores = output_dict["detection_scores"].numpy()[0]  # Confidence scores
        classes = output_dict["detection_classes"].numpy()[0].astype(int)  # Class labels

        height, width, _ = image.shape
        detections = []

        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                y_min, x_min, y_max, x_max = boxes[i]
                x_min, x_max = int(x_min * width), int(x_max * width)
                y_min, y_max = int(y_min * height), int(y_max * height)

                detections.append({
                    "box": [x_min, y_min, x_max, y_max],
                    "score": float(scores[i]),
                    "class": int(classes[i])
                })
                class_id = int(classes[i])
                class_name = COCO_CLASSES.get(class_id, "Unknown")
                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw label
                label = f"Class {class_name} ({scores[i]:.2f})"
                cv2.putText(image, label, (x_min, max(y_min - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode image as JPEG
        _, encoded_image = cv2.imencode(".jpg", image)

        # Return processed image as response
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg",
                        headers={"Content-Disposition": 'inline; filename="result.jpg"'})

    except Exception as e:
        return {"error": str(e)}
