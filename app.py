from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import time
import tempfile
import requests

# Initialize FastAPI app
app = FastAPI(title="Vehicle Installation Analysis API", version="1.0.0")

hf_token = "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"
client = None

@app.on_event("startup")
async def startup_event():
    global client
    try:
        from gradio_client import Client
        client = Client("Qwen/Qwen2.5-VL-32B-Instruct", hf_token=hf_token)
        print("✅ Gradio client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Gradio client: {e}")
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

# Accept list of image URLs
class ImageURLs(BaseModel):
    image_urls: list[HttpUrl]

@app.post("/analyze/")
async def analyze_images(payload: ImageURLs):
    if not client:
        return JSONResponse(
            content={"error": "Gradio client not initialized. Please check server logs."}, 
            status_code=500
        )

    temp_files = []
    try:
        from gradio_client import handle_file

        # Step 1: Download all images
        for idx, url in enumerate(payload.image_urls):
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
            print(f"📥 Downloaded image from {url} to {temp_file.name}")

        # Step 2: Upload all images to the model
        history = []
        for image_path in temp_files:
            history = client.predict(
                history=history,
                file=handle_file(image_path),
                api_name="/add_file"
            )
            print(f"✅ Uploaded image {image_path}")

        time.sleep(2)

        # Step 3: Add the text prompt
        prompt = ''' आपको एक ही वाहन की कई इमेज दी गई हैं, जो इंस्टॉलर द्वारा किए गए GPS डिवाइस इंस्टॉलेशन को दर्शाती हैं। इंस्टॉलर ने इंस्टॉलेशन पूरा करने के बाद डेटा अपलोड किया है, और अब आपको इसकी समीक्षा करनी है।

        आपका कार्य निम्नलिखित है:

        ✅ इंस्टॉलेशन सुधार बिंदु की सूची बनाएं जिसमें प्रत्येक बिंदु 5 शब्दों से अधिक न हो और उसके बाद "➡ Rating: X/10" के रूप में रेटिंग हो।

        हर बिंदु एक नंबर से दर्शाएं (1️⃣, 2️⃣, 3️⃣ …)।
        बिंदुओं जैसे "तार ढिले हैं – अच्छे से बांधो", "बहुत सारे तार बाहर हैं – टेप से लपेटो" आदि को शामिल करें।
        कुल इंस्टॉलेशन की रेटिंग दें, उदाहरण के तौर पर: "🚀 कुल इंस्टॉलेशन रेटिंग: 3/10"।
        अंत में एक छोटा प्रेरणादायक नोट दें, जैसे:
        ➡ अधिक रुपये चाहिए? इंस्टॉलेशन सुधारो, फिर अधिक मिलेगा।
        '''
        history = client.predict(
            text=prompt,
            api_name="/add_text"
        )
        print("✅ Text prompt added")

        time.sleep(3)

        # Step 4: Get the model's response
        result = client.predict(
            _chatbot=history,
            api_name="/predict"
        )

        # Clean up downloaded images
        for path in temp_files:
            os.unlink(path)

        if result and len(result) > 0:
            model_response = result[-1][1]
            return JSONResponse(content={
                "status": "success",
                "analysis": model_response,
                "image_count": len(payload.image_urls)
            })
        else:
            return JSONResponse(
                content={"error": "No response received from the model"}, 
                status_code=500
            )

    except Exception as e:
        for path in temp_files:
            try: os.unlink(path)
            except: pass
        print(f"❌ Error: {e}")
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, 
            status_code=500
        )
