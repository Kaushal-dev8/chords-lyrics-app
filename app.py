from flask import Flask, render_template, request
from analyzer import (
    get_chromagram,
    run_chord_recognition,
    identify_scale,
    align_chords_with_lyrics
)
from downloader import download_audio
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download", methods=["POST"])
def download():

    try:
        url = request.form.get("url")
        lyrics_text = request.form.get("lyrics") or ""

        if not url:
            return "<h1>Error</h1><p>No URL provided.</p><a href='/'>Try again</a>"

        # ------------------------
        # Download Audio
        # ------------------------
        file_path = download_audio(url)

        # Safety check
        if not file_path or not os.path.exists(file_path):
            return "<h1>Error</h1><p>Audio download failed.</p><a href='/'>Try again</a>"

        # ------------------------
        # Audio Processing
        # ------------------------
        chroma, sr, y_harmonic = get_chromagram(file_path)

        chords = run_chord_recognition(chroma, sr, y_harmonic)

        detected_key = identify_scale(chroma)

        # ------------------------
        # Align Lyrics
        # ------------------------
        if lyrics_text.strip():
            aligned = align_chords_with_lyrics(chords, lyrics_text)
        else:
            aligned = []

        return render_template(
            "results.html",
            scale=detected_key,
            aligned=aligned
        )

    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><a href='/'>Try again</a>"

if __name__ == "__main__":
    app.run(debug=True)
