import os
import subprocess
from flask import Flask, request, send_file
from flask_cors import CORS
from uuid import uuid4

app = Flask(__name__)
CORS(app)

@app.route("/ask-hal", methods=["POST"])
def ask_hal():
    audio_file = request.files["audio"]
    input_id = str(uuid4())
    input_path = f"/tmp/{input_id}.webm"
    wav_path = f"/tmp/{input_id}.wav"
    txt_path = f"/tmp/{input_id}.txt"

    # Save the uploaded audio
    audio_file.save(input_path)

    # Convert .webm to .wav
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        wav_path
    ], check=True)

    # Transcribe with Whisper.cpp
    subprocess.run([
        "./build/bin/whisper", "-m", "models/ggml-base.bin",
        "-f", wav_path, "-otxt", "-of", f"/tmp/{input_id}"
    ], check=True)

    # Read transcript
    with open(txt_path, "r") as f:
        text = f.read().strip()

    print("Transcribed:", text)

    # For now, just return the transcript text
    return text
