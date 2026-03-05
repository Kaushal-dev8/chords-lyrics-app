import yt_dlp
import os

def download_audio(url):

    # Ensure downloads folder exists
    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "downloads/%(title)s.%(ext)s",

        # Do NOT ignore errors anymore
        "nocheckcertificate": True,
        "no_warnings": True,
        "quiet": False,

        # Android client workaround
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],
                "player_js_version": ["actual"]
            }
        },

        "user_agent": "com.google.android.youtube/19.29.37 (Linux; U; Android 11; en_US)",

        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(url, download=True)

            # If extraction failed
            if info is None:
                return None

            filename = ydl.prepare_filename(info)

            base, _ = os.path.splitext(filename)
            final_path = f"{base}.wav"

            # Ensure file actually exists
            if os.path.exists(final_path):
                return final_path
            else:
                return None

    except Exception as e:
        print("Download error:", e)
        return None
