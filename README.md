# 📚 Smart Librarian

RAG (ChromaDB) + OpenAI GPT that recommends books from a local dataset and auto‑appends a **detailed summary** via tool‑calling.
Extras: **upload STT** (Whisper), **Live voice** via OpenAI Realtime (WebRTC), optional **TTS** and **symbolic cover**.

---

## ✨ Features

* **RAG** over `data/book_summaries.json` using OpenAI embeddings
* **Chatbot** (Streamlit): recommendation + context + full summary (tool call)
* **Voice→Text (upload)** via Whisper (batch)
* **Live Voice→Text** via OpenAI **Realtime** (WebRTC)
* Optional **TTS** (pyttsx3) & **image cover**

---

## 🗂 Project Structure

```
smart-librarian/
├─ app/
│  ├─ ui_streamlit.py         # Streamlit UI (main app)
│  ├─ init_vector_store.py    # builds Chroma DB from JSON
│  ├─ rag.py                  # embeddings + semantic search
│  ├─ chatbot.py              # chat logic + tool-calling
│  ├─ tools.py                # get_summary_by_title()
│  └─ speech.py               # STT (upload) + TTS helpers
├─ data/
│  ├─ book_summaries.json     # 50+ entries (title/author/year/genres/themes/summary)
│  └─ book_summaries.md       # (optional notes)
├─ token_server.py            # ephemeral token server for Realtime
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```

---

## ✅ Prerequisites

```bash
# Python 3.11 recommended
python --version

# FFmpeg (for Whisper); Windows:
winget install -e --id Gyan.FFmpeg

# Git (to clone/push)
git --version
```

---

## ⚙️ Setup

```bash
# 1) Clone
git clone https://github.com/<you>/smart-librarian.git
cd smart-librarian

# 2) Virtual env (Windows)
python -m venv .venv
. .venv/Scripts/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Env file (create from template)
copy .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

Key `.env` entries:

```bash
# .env (edit these)
OPENAI_API_KEY=sk-...

OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-small
CHAT_TEMPERATURE=0.3

REALTIME_MODEL=gpt-4o-mini-realtime-preview
REALTIME_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe

DATA_JSON=./data/book_summaries.json
CHROMA_DIR=./chroma
CHROMA_COLLECTION=books
RESET_COLLECTION=true
RAG_TOP_K=5
```

---

## 🧱 Build / Rebuild the Vector Store

```bash
# Optional: clean old DB
rmdir /S /Q .\chroma  # (Windows)

# Build embeddings into Chroma
python -m app.init_vector_store
# Expect: "Upserting 50 items ... DONE"
```

---

## ▶️ Run the App (Streamlit)

```bash
streamlit run app/ui_streamlit.py
# Open http://localhost:8501
```

Tabs:

* **Recomandare (text)** – type your query, get recommendation + full summary
* **Voice→Text (fișier)** – upload .wav/.mp3 → transcribe → search/recommend
* **Live (OpenAI Realtime)** – mic streaming with live transcription

---

## 🎙️ Live Voice (OpenAI Realtime)

```bash
# Run in a second terminal (same venv)
uvicorn token_server:app --port 5050 --reload
```

Then in the app → **Live (OpenAI Realtime)** tab:

1. **Start** → speak → text appears live
2. **Folosește ultima transcriere** → use it for RAG / recommendation

---

## 🧰 Tool (get\_summary\_by\_title)

Quick test from shell:

```bash
python - << 'PY'
from app.tools import get_summary_by_title
print(get_summary_by_title("Harry Potter and the Sorcerer's Stone")[:200], "…")
PY
```

* Reads from `DATA_JSON`
* Works with both formats: list of objects (current) and legacy dict

---

## 🧠 Data Format (RAG)

```json
[
  {
    "title": "Pride and Prejudice",
    "author": "Jane Austen",
    "year": 1813,
    "genres": ["Classic", "Romance"],
    "themes": ["love vs social class", "misjudgment", "family"],
    "summary": "Elizabeth Bennet clashes with the aloof Mr. Darcy in a comedy of manners about love and social expectations."
  }
]
```

> **Tip:** To improve niche queries (e.g., “mitologie nordică”), add a few representative titles and include those keywords naturally in **summary**, **genres**, or **themes**. Then rebuild the vector store.

---

## 🔐 Security & Git Hygiene

```bash
# Ensure .env isn’t tracked
git ls-files | findstr /I "\.env$"

# Scan repo for hardcoded keys
git grep -n -I "sk-"
git grep -n -I OPENAI_API_KEY
```

`.gitignore` already excludes `.env`, `chroma/`, temp assets (audio, images), etc.

---

## 🛠️ Troubleshooting

```bash
# Streamlit closes instantly ("Stopping…")
# → There’s a startup import error. Run from project ROOT and ensure you didn't import removed modules.

# NotFoundError: Collection [...] does not exists
# → .env mismatch. Set CHROMA_COLLECTION=books and rebuild:
rmdir /S /Q .\chroma
python -m app.init_vector_store

# ValueError: Expected metadata value ... got list
# → Use current init_vector_store.py (joins lists to strings) and rebuild.

# OpenAI 401
# → Set OPENAI_API_KEY in .env and restart Streamlit + token server.

# Whisper decode errors
# → Install FFmpeg, restart shell, retry.
```

---

## 📜 License

MIT (or your preferred license).

---

## 🙌 Credits

Built with **Streamlit**, **ChromaDB**, and **OpenAI** (chat, embeddings, Realtime, Whisper).
UI styled with a small custom CSS layer.
