from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
import os
import time
import json

# Set your Hugging Face token
hf_token = "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"  # Replace with your actual token

# Initialize client
client = Client("Qwen/Qwen2.5-VL-32B-Instruct", hf_token=hf_token)

app = FastAPI()

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Step 1: Upload the image
        history_after_image = client.predict(
            history=[],
            file=handle_file(image_path),
            api_name="/add_file"
        )

        # Add delay to ensure processing
        time.sleep(3)

        # Define your prompt
        prompt = ''' आपको एक ही वाहन की कई इमेज दी गई हैं, जो इंस्टॉलर द्वारा किए गए GPS डिवाइस इंस्टॉलेशन को दर्शाती हैं। इंस्टॉलर ने इंस्टॉलेशन पूरा करने के बाद डेटा अपलोड किया है, और अब आपको इसकी समीक्षा करनी है।

        आपका कार्य निम्नलिखित है:

        ✅ इंस्टॉलेशन सुधार बिंदु की सूची बनाएं जिसमें प्रत्येक बिंदु 5 शब्दों से अधिक न हो और उसके बाद “➡ Rating: X/10” के रूप में रेटिंग हो।

        हर बिंदु एक नंबर से दर्शाएं (1️⃣, 2️⃣, 3️⃣ …)।
        बिंदुओं जैसे “तार ढिले हैं – अच्छे से बांधो”, “बहुत सारे तार बाहर हैं – टेप से लपेटो” आदि को शामिल करें।
        कुल इंस्टॉलेशन की रेटिंग दें, उदाहरण के लिए: “🚀 कुल इंस्टॉलेशन रेटिंग: 3/10”।
        अंत में एक छोटा प्रेरणादायक नोट दें, जैसे:
        ➡ अधिक रुपये चाहिए? इंस्टॉलेशन सुधारो, फिर अधिक मिलेगा।
        '''

        # Step 2: Add the text prompt
        history_after_text = client.predict(
            text=prompt,
            api_name="/add_text"
        )

        # Add delay to ensure processing
        time.sleep(3)

        # Step 3: Get the model's response
        result = client.predict(
            _chatbot=history_after_text,
            api_name="/predict"
        )

        # Extract the final response
        if result and len(result) > 0:
            model_response = result[-1][1]
            return JSONResponse(content={"analysis": model_response})
        else:
            return JSONResponse(content={"error": "No response received from the model"}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

