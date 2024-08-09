from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from io import BytesIO

app = Flask(__name__)

# Configurez votre clé API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(file_stream):
    """Télécharge le fichier sur Gemini et retourne son URI."""
    try:
        # Téléchargez le fichier sur Gemini
        uploaded_file = genai.upload_file(file_stream, mime_type="image/jpg")
        return uploaded_file.uri
    except Exception as e:
        print(f"Erreur lors du téléchargement du fichier : {e}")
        return str(e)

@app.route('/', methods=['POST'])
def chat():
    try:
        file_uri = None
        if 'file' in request.files:
            file = request.files['file']
            file_bytes = file.read()
            file_stream = BytesIO(file_bytes)
            file_uri = upload_to_gemini(file_stream)

        data = request.form
        prompt = data.get('prompt', 'hello')
        custom_id = data.get('customId', 'default_id')

        # Créez l'historique de la conversation
        history = [
            {
                "role": "user",
                "parts": [file_uri, prompt] if file_uri else [prompt],
            }
        ]

        # Créez le modèle et démarrez la session de chat
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(prompt)

        return jsonify({"message": response.text, "customId": custom_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
