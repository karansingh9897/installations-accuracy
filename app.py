from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import aiohttp
import os
import time
from typing import List

from gradio_client import Client, handle_file

# Initialize FastAPI
app = FastAPI(title="Vehicle Installation Reviewer API", version="1.0")

# Hugging Face token (replace if needed)
hf_token = "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"

# Load Gradio Client on startup
client = None

@app.on_event("startup")
async def load_model():
    global client
    try:
        client = Client("Qwen/Qwen2.5-VL-32B-Instruct", hf_token=hf_token)
        print("✅ Gradio client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize client: {str(e)}")
        client = None

# Request body schema
class AnalyzeRequest(BaseModel):
    image_urls: List[str]
    prompt: str

# Endpoint
@app.post("/analyze/")
async def analyze_images(request: AnalyzeRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gradio client not initialized")

    temp_files = []
    try:
        print("📥 Downloading images...")
        # Download all images
        async with aiohttp.ClientSession() as session:
            for url in request.image_urls:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download image: {url}")
                    content = await response.read()
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    temp_file.write(content)
                    temp_file.close()
                    temp_files.append(temp_file.name)

        print("✅ Images downloaded and saved.")

        # Upload images one by one
        history = []
        for path in temp_files:
            history = client.predict(
                history=history,
                file=handle_file(path),
                api_name="/add_file"
            )
        print("✅ Images uploaded to model.")

        # Wait for model to process the images
        time.sleep(2)

        # Add the prompt
        history = client.predict(
            text=request.prompt,
            api_name="/add_text"
        )
        print("✅ Prompt sent to model.")

        time.sleep(2)

        # Get the model output
        result = client.predict(
            _chatbot=history,
            api_name="/predict"
        )
        print("✅ Received response from model.")

        # Parse final result
        if result and isinstance(result, list):
            final_response = result[-1][1]
            return JSONResponse(content={
                "status": "success",
                "image_count": len(request.image_urls),
                "analysis": final_response
            })
        else:
            raise HTTPException(status_code=500, detail="Unexpected model output format.")

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exception occurred: {str(e)}")

    finally:
        # Clean up downloaded image files
        for path in temp_files:
            if os.path.exists(path):
                os.unlink(path)
