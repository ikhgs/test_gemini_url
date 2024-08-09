import os
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Google Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Function to download image from URL or handle image upload
def save_uploaded_image(image):
    image_path = "temp_image.jpg"
    image.save(image_path)
    return image_path

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route('/gemini', methods=['POST'])
def gemini():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']
    prompt = request.form.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Save the uploaded image
    image_path = save_uploaded_image(image)
    file = upload_to_gemini(image_path, mime_type="image/jpeg")

    # Create the chat session with the image and text prompt
    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    ).start_chat(
        history=[
            {
                "role": "user",
                "parts": [file, prompt],
            }
        ]
    )

    response = chat_session.send_message(prompt)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
