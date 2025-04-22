import os
from flask import Flask, request, send_file
from flask_cors import CORS
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

@app.route("/ask-hal", methods=["POST"])
def ask_hal():
    # Save uploaded audio file
    audio_file = request.files["audio"]
    input_path = f"/tmp/{uuid4()}.webm"
    audio_file.save(input_path)

    # Transcribe using OpenAI
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(input_path, "rb")
    )
    text = transcript.text
    print("Transcribed:", text)

    # Generate response using OpenAI
    chat = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{
            "role": "system",
            "content": "You are HAL 9000. Be calm, polite, eerie, confident."
        }, {
            "role": "user",
            "content": text
        }]
    )
    hal_reply = chat.choices[0].message.content
    print("HAL says:", hal_reply)

    # Convert reply to audio using ElevenLabs
    voice = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
    out_path = f"/tmp/{uuid4()}.mp3"

    audio = voice.text_to_speech.convert(
        voice_id=os.getenv("ELEVEN_VOICE_ID"),
        model_id="eleven_multilingual_v2",
        optimize_streaming_latency=4,
        input=hal_reply
    )

    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return send_file(out_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
