from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from deepface import DeepFace
from starlette.responses import JSONResponse

app = FastAPI()

@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Convert uploaded image to OpenCV format
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Load OpenCV face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]  # Extract face region

            # Run DeepFace Analysis
            analysis = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
            results.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "age": analysis["age"],
                "gender": analysis["dominant_gender"],
                "emotion": analysis["dominant_emotion"]
            })

        return JSONResponse(content={"faces": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
