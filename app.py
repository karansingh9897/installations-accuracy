from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import tempfile
import requests
import time
import os

app = FastAPI(title="Vehicle Installation Analysis API", version="1.1.0")

hf_token = "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"
client = None

# Initialize Gradio client on startup
@app.on_event("startup")
async def startup_event():
    global client
    try:
        from gradio_client import Client
        client = Client("Qwen/Qwen2.5-VL-32B-Instruct", hf_token=hf_token)
        print("✅ Gradio client initialized successfully")
    except Exception as e:
        print(f"❌ Gradio client init error: {e}")
        client = None

@app.get("/")
async def root():
    return {"message": "Vehicle Installation Analysis API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "client_status": "connected" if client else "disconnected"
    }

# ✅ Define payload schema
class ImagePromptRequest(BaseModel):
    image_urls: list[HttpUrl]
    prompt: str

@app.post("/analyze/")
async def analyze_images_with_prompt(request: ImagePromptRequest):
    if not client:
        return JSONResponse(
            content={"error": "Gradio client not initialized."}, 
            status_code=500
        )

    temp_files = []
    try:
        from gradio_client import handle_file

        history = []

        # 1️⃣ Download and upload all images
        for idx, url in enumerate(request.image_urls):
            response = requests.get(url)
            if response.status_code != 200:
                return JSONResponse(
                    content={"error": f"Failed to download image at index {idx}: {url}"},
                    status_code=400
                )

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(response.content)
            temp_file.close()
            temp_files.append(temp_file.name)

            history = client.predict(
                history=history,
                file=handle_file(temp_file.name),
                api_name="/add_file"
            )
            print(f"✅ Uploaded image {idx+1}/{len(request.image_urls)}")

        # 2️⃣ Add user prompt
        history = client.predict(
            text=request.prompt,
            api_name="/add_text"
        )
        print("✅ Prompt submitted")

        time.sleep(2)

        # 3️⃣ Get model's final response
        result = client.predict(
            _chatbot=history,
            api_name="/predict"
        )

        for path in temp_files:
            try: os.unlink(path)
            except: pass

        if result and len(result) > 0:
            return JSONResponse(content={
                "status": "success",
                "image_count": len(request.image_urls),
                "analysis": result[-1][1]
            })
        else:
            return JSONResponse(
                content={"error": "No response from model."}, 
                status_code=500
            )

    except Exception as e:
        for path in temp_files:
            try: os.unlink(path)
            except: pass
        return JSONResponse(
            content={"error": f"Exception occurred: {str(e)}"},
            status_code=500
        )
