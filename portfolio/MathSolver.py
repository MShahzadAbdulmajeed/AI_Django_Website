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
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer, pipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

p2t = Pix2Text.from_config()
model_name_or_path = "D:/huggingface_models/qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="auto",  # Let accelerate handle device placement
    torch_dtype=torch.float16 if device == "cuda:0" else torch.float32
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_length=512,  
    repetition_penalty=1.1
)
custom_pipe = HuggingFacePipeline(pipeline=pipe)

# Define the prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["problem"],
    template="Solve the following equation step by step: {problem}"
)

# Set up LLMChain with the custom pipeline
llm_chain = LLMChain(
    llm=custom_pipe,
    prompt=prompt_template
)
def extract_math_equation(image):
    image = image.resize((320, 768))  # Resize to avoid warning
    latex_output = p2t.recognize(image, file_type="text_formula", return_text=True)
    latex_equation = " ".join(latex_output) if isinstance(latex_output, list) else latex_output
    latex_equation = re.sub(r"\s+", " ", latex_equation).strip()
    print(f"Extracted LaTeX: {latex_equation}")  # Debugging print
    print("extrac math equation run")
    return latex_equation


def solve_math_problem(latex_equation):
    
    
    response = llm_chain.run({"problem": latex_equation})
    
    answer = response.split("Answer:")[-1].strip()

    
    return answer
    
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
