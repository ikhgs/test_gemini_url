import os
import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Google Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Function to download image from URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        filename = "temp_image.jpg"
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    return None

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route('/gemini', methods=['POST'])
def gemini():
    data = request.json
    prompt = data.get('prompt')
    image_url = data.get('image_url')

    # Download image from URL
    if image_url:
        image_path = download_image(image_url)
        if image_path:
            file = upload_to_gemini(image_path, mime_type="image/jpeg")
            files = [file]
        else:
            return jsonify({"error": "Failed to download image"}), 400
    else:
        files = []

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
                "parts": files + [prompt],
            }
        ]
    )

    response = chat_session.send_message(prompt)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
