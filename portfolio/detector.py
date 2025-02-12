import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import kagglehub

# Download and load the Faster R-CNN Inception ResNet V2 model
MODEL_PATH = kagglehub.model_download("tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/640x640")
model = tf.saved_model.load(MODEL_PATH)

# COCO class labels mapping
COCO_CLASSES = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane", 6: "Bus", 7: "Train", 8: "Truck",
    9: "Boat", 10: "Traffic Light", 11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 21: "Cow", 22: "Elephant", 23: "Bear",
    24: "Zebra", 25: "Giraffe", 27: "Backpack", 28: "Umbrella", 31: "Handbag", 32: "Tie", 33: "Suitcase",
    34: "Frisbee", 35: "Skis", 36: "Snowboard", 37: "Sports Ball", 38: "Kite", 39: "Baseball Bat",
    40: "Baseball Glove", 41: "Skateboard", 42: "Surfboard", 43: "Tennis Racket", 44: "Bottle", 46: "Wine Glass",
    47: "Cup", 48: "Fork", 49: "Knife", 50: "Spoon", 51: "Bowl", 52: "Banana", 53: "Apple", 54: "Sandwich",
    55: "Orange", 56: "Broccoli", 57: "Carrot", 58: "Hot Dog", 59: "Pizza", 60: "Donut", 61: "Cake",
    62: "Chair", 63: "Couch", 64: "Potted Plant", 65: "Bed", 67: "Dining Table", 70: "Toilet", 72: "TV",
    73: "Laptop", 74: "Mouse", 75: "Remote", 76: "Keyboard", 77: "Cell Phone", 78: "Microwave", 79: "Oven",
    80: "Toaster", 81: "Sink", 82: "Refrigerator", 84: "Book", 85: "Clock", 86: "Vase", 87: "Scissors",
    88: "Teddy Bear", 89: "Hair Drier", 90: "Toothbrush"
}


def preprocess_image(image_path):
    """Load and preprocess the input image."""
    img = Image.open(image_path).convert("RGB")
    img = np.array(img, dtype=np.uint8)  # Convert to uint8 NumPy array
    img = tf.image.resize(img, (640, 640))  # Resize to model's expected input size
    img = tf.cast(img, dtype=tf.uint8)  # Ensure dtype is uint8
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def run_inference(image_path):
    """Run object detection inference on an image."""
    input_tensor = preprocess_image(image_path)
    infer = model.signatures["serving_default"]
    detections = infer(input_tensor)

    boxes = detections["detection_boxes"].numpy()[0]  # Bounding boxes
    scores = detections["detection_scores"].numpy()[0]  # Confidence scores
    classes = detections["detection_classes"].numpy()[0].astype(int)  # Class IDs

    return boxes, scores, classes

def visualize(image_path, boxes, scores, classes, threshold=0.5):
    """Draw bounding boxes and class labels on the image."""
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    height, width, _ = image.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

            class_id = classes[i]
            class_name = COCO_CLASSES.get(class_id, "Unknown")

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    display_width = 800
    aspect_ratio = display_width / width
    new_height = int(height * aspect_ratio)
    resized_img = cv2.resize(image, (display_width, new_height))

    return resized_img