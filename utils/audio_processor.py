import os
import tempfile

import yt_dlp
from pydub import AudioSegment

# Absolute path outside OneDrive: no sync locks, same place however you launch the app
DOWNLOAD_DIR = os.environ.get("VIDEO_AGENT_DIR", os.path.join(tempfile.gettempdir(), "ai_video_agent"))
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# Downloading youtube audio from link
def download_youtube_audio(url: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        # video id instead of title: no emoji / special characters in filenames
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # mono + 16 kHz (what Whisper uses anyway): ~8x smaller than the default
        "postprocessor_args": {"extractaudio": ["-ar", "16000", "-ac", "1"]},
        # False shows the real ffmpeg error if conversion fails; set True once it works
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Ask yt-dlp for the final file instead of guessing the extension
        downloads = info.get("requested_downloads") or []
        if downloads and downloads[0].get("filepath"):
            return downloads[0]["filepath"]
        return os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"


# Converting into wav format
def convert_to_wav(input_path: str) -> str:
    """Convert any raw audio/video file into mono 16 kHz wav using pydub."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(DOWNLOAD_DIR, f"{base}_converted.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


# Chunking
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_file(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    stem = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{stem}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    # Strip whitespace and the quotes Windows adds with "Copy as path"
    source = source.strip().strip('"').strip("'")

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected youtube URL, downloading audio..")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file, converting into wav format")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunks created")
    return chunks