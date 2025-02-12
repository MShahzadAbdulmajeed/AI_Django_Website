from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import uvicorn
from fastapi.responses import Response
import base64
from fastapi.responses import JSONResponse
app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a different model if needed (e.g., yolov8s.pt)

def preprocess_image(image_bytes):
    """Convert image bytes to OpenCV format."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/live-object-detection/")
async def predict(file: UploadFile = File(...)):
    """Perform object detection on the uploaded image."""
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    # Run inference
    results = model(img)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence,
                "class": model.names[class_id]
            })
            color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _, img_encoded = cv2.imencode(".jpg", img)
    image_bas64 = base64.b64encode(img_encoded).decode("utf-8")
    return JSONResponse({"detections": detections, "image": image_bas64})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
