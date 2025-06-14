from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from gradio_client import Client, handle_file
import os
import time
import json
import traceback

app = FastAPI()

# Set your Hugging Face API Token
hf_token = "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"

# Initialize Qwen Client
client = Client("Qwen/Qwen2.5-VL-32B-Instruct", hf_token=hf_token)

def generate_report(image_urls, prompt):
    """Helper function to generate vehicle installation report from Qwen API."""
    try:
        # 1. Upload all images first
        for image_url in image_urls:
            history_after_file = client.predict(
                history=[], 
                file=handle_file(image_url), 
                api_name="/add_file"
            )
        time.sleep(3)

        # 2. Send prompt afterwards
        history_after_prompt = client.predict(
            text=prompt,
            api_name="/add_text"
        )
        time.sleep(3)

        # 3. Finally, generate the response
        result = client.predict(
            _chatbot=history_after_prompt,
            api_name="/predict"
        )
        print("Qwen API raw result:")
        print(result)
        
        if result and len(result) > 0:
            return result[-1][1]
        else:
            return "Model did not return a response."

    except Exception as e:
        return f"Error during API call: {str(e)}\n{traceback.format_exc()}"

# Define pydantic model for input
class ReportRequest(BaseModel):
    images: List[str]
    prompt: str

@app.post("/generate_report")
def generate_report_endpoint(data: ReportRequest):
    """API endpoint to generate vehicle installation report."""
    images = data.images
    prompt = data.prompt
    
    if not images or not isinstance(images, list):
        raise HTTPException(400, "images must be a non-empty list of URLs.")
    if not prompt or not isinstance(prompt, str):
        raise HTTPException(400, "prompt must be a non-empty string.")
    
    report = generate_report(images, prompt)

    try:
        clean_json = report.split('
json')[1].strip('`\n')

        parsed = json.loads(clean_json)
        return parsed
    except Exception as e:
        raise HTTPException(500, f"Failed to parse Qwen's response: {str(e)}")
