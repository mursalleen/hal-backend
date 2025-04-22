import os
import tempfile
from flask import Flask, request, send_file
from flask_cors import CORS
import soundfile as sf
import requests
import openai
from elevenlabs import clone, generate
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")
set_api_key(os.getenv("ELEVEN_API_KEY"))
VOICE = os.getenv("ELEVEN_VOICE_ID")


@app.route("/ask-hal", methods=["POST"])
def ask_hal():
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.read())
        f.flush()

        # Transcribe
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", file=open(f.name, "rb")
        )
        prompt = transcript.text

        # Get AI response
        completion = openai.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are HAL 9000 from 2001: A Space Odyssey. Keep responses calm, emotionless, short, and psychologically provocative.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        reply = completion.choices[0].message.content

        # TTS via ElevenLabs
        audio = generate(
            text=reply,
            voice=VOICE,
            model="eleven_multilingual_v1"
        )
        output_path = os.path.join(tempfile.gettempdir(), "hal-response.wav")
        with open(output_path, "wb") as f:
            f.write(audio)

    return send_file(output_path, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
