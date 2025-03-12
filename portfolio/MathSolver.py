from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import Optional
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
import torch

from pix2text import Pix2Text
import re
from fastapi.responses import JSONResponse
from transformers import BitsAndBytesConfig
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer

# ✅ Detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Pix2Text for OCR
p2t = Pix2Text.from_config()

# ✅ Load DeepSeek-Math-7B Model from Local Path
local_model_path = "D:/huggingface_models/qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 🔥 Ultra-fast 4-bit mode
    bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ Optimized for modern GPUs
    bnb_4bit_use_double_quant=True,  # ✅ Extra compression
)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    low_cpu_mem_usage=True,
    device_map="cuda",  # Automatically distributes model across available devices
    torch_dtype=torch.float16 if device == "cuda:0" else torch.float32
)





# ✅ Function to clean and format the response for human readability
# def clean_math_response(response: str) -> str:
#     """ Cleans a raw LaTeX-based math response to make it human-readable. """
#     response = re.sub(r'^.*?(Problem:)', r'\1', response, flags=re.DOTALL)
#     response = re.sub(r'\$\$', '', response)  # Remove LaTeX equation markers
#     response = response.replace('\\\\', '\n')  # Remove excessive backslashes
#     response = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', response)  # Convert fractions
#     response = re.sub(r'\\textbf\{(.*?)\}', r'\1', response)  # Remove \textbf
#     response = response.replace('-(-', '+').replace(')', '')  # Fix double negatives
#     response = re.sub(r'\\boxed\{(.*?)\}', r'Answer: \1', response)  # Format boxed answers
#     response = re.sub(r"[^\x00-\x7F]+", "", response)  # Remove non-ASCII characters
#     response = re.sub(r"Solution:?", "", response).strip()  # Remove extra "Solution:"
#     return response

# ✅ Extract Math Equation from Image
def extract_math_equation(image):
    image = image.resize((320, 768))  # Resize to avoid warning
    latex_output = p2t.recognize(image, file_type="text_formula", return_text=True)
    latex_equation = " ".join(latex_output) if isinstance(latex_output, list) else latex_output
    latex_equation = re.sub(r"\s+", " ", latex_equation).strip()
    print(f"Extracted LaTeX: {latex_equation}")  # Debugging print
    return latex_equation

# ✅ Solve Math Problem
def solve_math_problem(latex_equation):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    system_message = "you are a world most intelligent ai that can solve math problems setp by step using math rules. your name is MathSolver."
    prompt = latex_equation
    prompt_template=f'''<|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    '''
    generation_params = {
    "do_sample": True,
    "temperature": 0.5,
    "top_p": 0.7,
    "top_k": 30,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    
    }
    tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
    ).input_ids.cuda()
    model.generate(
    tokens,
    streamer=streamer,
    **generation_params
    )

    # generation_thread = threading.Thread(
    #       target=model.generate,
    #       args=(tokens,),
    #       kwargs={**generation_params, 'streamer': streamer}
    #   )
    
    
    output = streamer.output()
    solution = tokenizer.decode(output, skip_special_tokens=True).strip()

    # generation_thread.start()
    return solution
    # Return the first generated output
    # raw_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    # cleaned_response = clean_math_response(raw_response)  # ✅ Apply the cleaning function
    # raw_response = re.sub(r".*?Problem:", "", cleaned_response, flags=re.DOTALL).strip()
    # return raw_response
# ✅ Create FastAPI App
app = FastAPI()

# ✅ Enable CORS for Django Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Handle Math Solving (Supports JSON and Form Data)
@app.post("/solve_math")
async def process_math_request(request: Request):
    if request.headers.get("content-type") == "application/json":
        json_data = await request.json()
        user_text = json_data.get("user_text")

        if not user_text:
            return {"error": "No math equation provided"}

        solution = solve_math_problem(user_text)

        return {"problem": user_text, "solution": solution}

    return {"error": "Invalid content-type. Use application/json"}

# ✅ Start the FastAPI Server
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
