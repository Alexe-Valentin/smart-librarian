import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

# ----- TTS (pyttsx3: fully offline) -----
def tts_say(text: str, out_dir: str | Path, filename: str = "recommendation.wav") -> Optional[str]:
    """Speak `text` to a WAV file and return its path, or None on failure."""
    try:
        import pyttsx3
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        engine = pyttsx3.init()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return str(out_path)
    except Exception:
        return None

# ----- STT (openai-whisper: needs torch) -----
def transcribe_audio(input_path: str | Path) -> str:
    """
    Transcribe audio to text using openai-whisper (CPU).
    Accepts WAV/MP3; for MP3 needs ffmpeg installed.
    """
    import whisper
    model_name = os.getenv("STT_MODEL", "base")  # tiny, base, small, medium, large
    model = whisper.load_model(model_name)
    result = model.transcribe(str(input_path))
    return result.get("text", "").strip()
