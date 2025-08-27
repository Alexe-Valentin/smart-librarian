# app/init_vector_store.py
"""
(Re)build the ChromaDB vector store from data/book_summaries.json.

- Expects each item to have: title, author, year, genres[list], themes[list], summary[str]
- Embeds a rich text:  "{title}\n{summary}\nGenres: ...\nThemes: ..."
- Stores the summary as the document (nice for snippets)
- Metadata must be scalars => genres/themes saved as comma-separated strings
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI
import chromadb

# -------------------- Env & constants --------------------

DATA_JSON = os.getenv("DATA_JSON", "./data/book_summaries.json")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "books")
EMBED_MODEL = os.getenv(
    "OPENAI_MODEL_EMBED", os.getenv("OPENAI_MODEL_EMBEDDINGS", "text-embedding-3-small")
)
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "true").lower() in {"1", "true", "yes", "y"}

client_oai = OpenAI()

# -------------------- Helpers ----------------------------

def _slug(s: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in s.strip())
    base = "-".join([p for p in base.split("-") if p])
    if not base:
        base = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return base[:64]

def _load_data(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATA_JSON file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("DATA_JSON must be a JSON array")

    out: List[Dict[str, Any]] = []
    for item in data:
        title = (item.get("title") or "").strip()
        author = (item.get("author") or "").strip()
        year = item.get("year")
        genres = item.get("genres") or []
        themes = item.get("themes") or []
        summary = (item.get("summary") or "").strip()

        if not title or not summary:
            continue

        if not isinstance(genres, list): genres = [str(genres)]
        if not isinstance(themes, list): themes = [str(themes)]
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None

        out.append({
            "title": title,
            "author": author,
            "year": year,
            "genres": [str(g).strip() for g in genres if str(g).strip()],
            "themes": [str(t).strip() for t in themes if str(t).strip()],
            "summary": summary
        })
    if not out:
        raise ValueError("DATA_JSON parsed but contains no valid items.")
    return out

def _compose_index_text(rec: Dict[str, Any]) -> str:
    genres = ", ".join(rec.get("genres", []))
    themes = ", ".join(rec.get("themes", []))
    return f"{rec['title']}\n{rec['summary']}\nGenres: {genres}\nThemes: {themes}".strip()

def _embed_batch(texts: List[str]) -> List[List[float]]:
    resp = client_oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# -------------------- Build collection -------------------

def build_collection():
    data = _load_data(DATA_JSON)

    # Build records
    seen: set[str] = set()
    records: List[Tuple[str, Dict[str, Any], str, str]] = []
    for i, rec in enumerate(data):
        rid = f"book-{i:03d}-{_slug(rec['title'])}"
        if rid in seen:
            rid = f"{rid}-{hashlib.sha1((rec['title']+str(i)).encode()).hexdigest()[:6]}"
        seen.add(rid)

        index_text = _compose_index_text(rec)
        document = rec["summary"]

        # IMPORTANT: Chroma metadata must be scalars; join lists to strings
        metadata = {
            "title": rec["title"],
            "author": rec["author"],
            "year": rec["year"],  # int or None is fine
            "genres": ", ".join(rec["genres"]),
            "themes": ", ".join(rec["themes"]),
        }
        records.append((rid, metadata, index_text, document))

    # Create Chroma client / collection
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    if RESET_COLLECTION:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Upsert in batches
    total = len(records)
    print(f"[init_vector_store] Upserting {total} items to collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    t0 = time.time()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = records[start:end]
        ids = [r[0] for r in batch]
        metadatas = [r[1] for r in batch]
        index_texts = [r[2] for r in batch]
        documents = [r[3] for r in batch]

        vectors = _embed_batch(index_texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"  • [{start:>3}-{end:>3}] upserted")

    dt = time.time() - t0
    print(f"[init_vector_store] DONE in {dt:.2f}s — {total} items.")

if __name__ == "__main__":
    build_collection()
