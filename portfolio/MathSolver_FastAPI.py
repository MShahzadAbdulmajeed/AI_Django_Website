import io
import logging
import onnxruntime as ort
import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from MathSolver import extract_math_equation, solve_math_problem  # Ensure these are implemented

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ CORS Settings (Allow Django & Localhost Requests)
origins = [
    "http://127.0.0.1:8000",  # Allow Django requests
    "http://localhost:8000",
    "http://127.0.0.1:8090",  # Existing Django Server
    "http://localhost:8090"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ONNX Model Loading with GPU (Fallback to CPU)
try:
    session = ort.InferenceSession("your_model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    logging.info("ONNX Model loaded successfully with CUDAExecutionProvider.")
except Exception as e:
    logging.error(f"Failed to load ONNX model: {e}")
    session = None  # Handle gracefully if model fails to load

# ✅ Pydantic Model for JSON Input
class MathRequest(BaseModel):
    user_text: Optional[str] = None

# ✅ FastAPI Route for Math Solver
@app.post("/solve-math/")
async def solve_math(
    request: Request,
    image: Optional[UploadFile] = File(None),
    user_text: Optional[str] = Form(None)  # Supports form-data text input
):
    try:
        # ✅ Extract JSON if request type is application/json
        if request.headers.get("content-type") == "application/json":
            json_data = await request.json()
            user_text = json_data.get("user_text")

        logging.info(f"Received text: {user_text}")
        logging.info(f"Received image: {image.filename if image else 'No Image'}")

        # ✅ Check if both inputs are missing
        if not image and not user_text:
            return JSONResponse(content={"error": "No valid input provided."}, status_code=400)

        # ✅ Process Image (if provided)
        if image:
            image_bytes = await image.read()  # Read image bytes
            image_pil = Image.open(io.BytesIO(image_bytes))  # Convert to PIL image
            latex_output = extract_math_equation(image_pil)  # Extract math equation
        else:
            latex_output = user_text  # Use text input directly

        logging.info(f"Extracted LaTeX: {latex_output}")

        # ✅ Solve Math Problem
        solution = solve_math_problem(latex_output)

        logging.info(f"Generated solution: {solution}")

        return {"latex": latex_output, "solution": solution}

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)

# ✅ Run FastAPI Server
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

# @app.post("/solve-math/")
# async def solve_math(
#     image: Optional[UploadFile] = File(None),
#     user_text: Optional[str] = Form(None)
# ):
#     # If both image and text are missing, return an error
#     if not image and not user_text:
#         return {"error": "Please provide either text or an image"}

#     # Process image if provided (in memory)
#     if image:
#         image_bytes = await image.read()  # Read image bytes
#         image_pil = Image.open(io.BytesIO(image_bytes))  # Open as PIL image
#         latex_output = extract_math_equation(image_pil)  # Process without saving
#     else:
#         latex_output = user_text  # Use user text if no image

#     # Solve the math problem
#     solution = solve_math_problem(latex_output)

#     return {"latex": latex_output, "solution": solution}

# if __name__ == "__main__":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)


    


