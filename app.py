from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import time
import tempfile

# Initialize FastAPI app first
app = FastAPI(title="Vehicle Installation Analysis API", version="1.0.0")

# Set your Hugging Face token
hf_token = "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"  # Replace with your actual token

# Initialize Gradio client after app creation
client = None

@app.on_event("startup")
async def startup_event():
    global client
    try:
        from gradio_client import Client, handle_file
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

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    if not client:
        return JSONResponse(
            content={"error": "Gradio client not initialized. Please check server logs."}, 
            status_code=500
        )
    
    try:
        from gradio_client import handle_file
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "Please upload an image file"}, 
                status_code=400
            )
        
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            image_path = temp_file.name

        print(f"📁 Image saved to: {image_path}")

        # Step 1: Upload the image
        history_after_image = client.predict(
            history=[],
            file=handle_file(image_path),
            api_name="/add_file"
        )
        print("✅ Image uploaded successfully")

        # Add delay to ensure processing
        time.sleep(3)

        # Define your prompt
        prompt = ''' आपको एक ही वाहन की कई इमेज दी गई हैं, जो इंस्टॉलर द्वारा किए गए GPS डिवाइस इंस्टॉलेशन को दर्शाती हैं। इंस्टॉलर ने इंस्टॉलेशन पूरा करने के बाद डेटा अपलोड किया है, और अब आपको इसकी समीक्षा करनी है।

        आपका कार्य निम्नलिखित है:

        ✅ इंस्टॉलेशन सुधार बिंदु की सूची बनाएं जिसमें प्रत्येक बिंदु 5 शब्दों से अधिक न हो और उसके बाद "➡ Rating: X/10" के रूप में रेटिंग हो।

        हर बिंदु एक नंबर से दर्शाएं (1️⃣, 2️⃣, 3️⃣ …)।
        बिंदुओं जैसे "तार ढिले हैं – अच्छे से बांधो", "बहुत सारे तार बाहर हैं – टेप से लपेटो" आदि को शामिल करें।
        कुल इंस्टॉलेशन की रेटिंग दें, उदाहरण के तौर पर: "🚀 कुल इंस्टॉलेशन रेटिंग: 3/10"।
        अंत में एक छोटा प्रेरणादायक नोट दें, जैसे:
        ➡ अधिक रुपये चाहिए? इंस्टॉलेशन सुधारो, फिर अधिक मिलेगा।
        '''

        # Step 2: Add the text prompt
        history_after_text = client.predict(
            text=prompt,
            api_name="/add_text"
        )
        print("✅ Text prompt added successfully")

        # Add delay to ensure processing
        time.sleep(3)

        # Step 3: Get the model's response
        result = client.predict(
            _chatbot=history_after_text,
            api_name="/predict"
        )
        print("✅ Model response generated")

        # Clean up temporary file
        os.unlink(image_path)

        # Extract the final response
        if result and len(result) > 0:
            model_response = result[-1][1]
            return JSONResponse(content={
                "status": "success",
                "analysis": model_response,
                "filename": file.filename,
                "file_size": len(content)
            })
        else:
            return JSONResponse(
                content={"error": "No response received from the model"}, 
                status_code=500
            )

    except Exception as e:
        # Clean up temporary file if it exists
        if 'image_path' in locals():
            try:
                os.unlink(image_path)
            except:
                pass
        
        print(f"❌ Error in analyze_image: {str(e)}")
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
