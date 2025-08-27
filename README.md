# 📚 Smart Librarian

RAG (ChromaDB) + OpenAI GPT: recommends a book from a local dataset and auto-fetches a detailed summary via tool-calling.  
Extras: upload STT (Whisper), **Live** voice via OpenAI Realtime (WebRTC), optional TTS and symbolic cover generation.

## Features
- Vector search (Chroma) over `data/book_summaries.json`
- Recommendation-only guardrails (no fiction writing)
- Tool-calling: `get_summary_by_title(title)`
- Upload audio → Whisper (batch STT)
- Live mic → OpenAI Realtime (WebRTC)
- Optional TTS (pyttsx3) + cover image (gpt-image-1)
- Preferences & logging (simple JSON/CSV)

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
copy .env.example .env     # fill OPENAI_API_KEY
python -m app.init_vector_store
