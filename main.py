from flask import Flask, request, send_file
from flask_cors import CORS
import os
import tempfile
from openai import OpenAI
from elevenlabs import generate, save
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.base_url = os.getenv("OPENAI_BASE_URL")
ELKEY = os.getenv("ELEVEN_API_KEY")
VOICE = os.getenv("ELEVEN_VOICE_ID")

@app.route("/ask-hal", methods=["POST"])
def ask_hal():
    print("🔴 Received request at /ask-hal")

    if "audio" not in request.files:
        return "No audio file found", 400

    audio_file = request.files["audio"]
    temp_path = tempfile.mktemp(suffix=".wav")
    audio_file.save(temp_path)

    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=open(temp_path, "rb")
    ).text

    print("User said:", transcript)

    response = openai.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[
            {"role": "system", "content": "You are HAL 9000. Respond briefly, calmly, and critically."},
            {"role": "user", "content": transcript}
        ]
    ).choices[0].message.content

    print("HAL:", response)

    audio = generate(
        text=response,
        voice=VOICE,
        model="eleven_monolingual_v1",
        api_key=ELKEY
    )

    out_path = tempfile.mktemp(suffix=".mp3")
    save(audio, out_path)

    return send_file(out_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
