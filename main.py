import os
from flask import Flask, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from uuid import uuid4
import subprocess

load_dotenv()
app = Flask(__name__)
CORS(app)

@app.route("/ask-hal", methods=["POST"])
def ask_hal():
    # Save uploaded audio
    audio_file = request.files["audio"]
    input_path = f"/tmp/{uuid4()}.webm"
    audio_file.save(input_path)

    # Convert to WAV for whisper.cpp
    wav_path = f"/tmp/{uuid4()}.wav"
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        wav_path
    ], check=True)

    # Transcribe using whisper.cpp
    result = subprocess.run([
        "./build/bin/whisper",
        "-m", "models/ggml-base.bin",
        "-f", wav_path,
        "-otxt", "-of", "/tmp/out"
    ], capture_output=True, text=True)

    # Read transcribed text
    with open("/tmp/out.txt", "r") as f:
        text = f.read().strip()
    print("User said:", text)

    # Reply with dummy message for now
    reply = f"I'm sorry, Dave. You said: {text}"

    # Save reply as mp3 (placeholder)
    reply_path = f"/tmp/{uuid4()}.mp3"
    subprocess.run([
        "say", reply, "-o", reply_path
    ], check=True)

    return send_file(reply_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
