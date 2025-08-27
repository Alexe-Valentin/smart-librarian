# app/rag.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv(override=True)

CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "books")  # <- default 'books'
EMBED_MODEL     = os.getenv(
    "OPENAI_MODEL_EMBED",
    os.getenv("OPENAI_MODEL_EMBEDDINGS", "text-embedding-3-small")
)

_client = OpenAI()
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(name=COLLECTION_NAME)  # robust

def embed(text: str) -> List[float]:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def search_books(query: str, k: int = 5) -> List[Dict[str, Any]]:
    vec = embed(query)
    res = _collection.query(query_embeddings=[vec], n_results=k,
                            include=["documents","metadatas","distances"])
    out: List[Dict[str, Any]] = []
    if res and res.get("ids"):
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i] or {}
            doc  = (res["documents"][0][i] or "").strip()
            dist = float(res["distances"][0][i])
            score = 1.0 / (1.0 + dist)
            out.append({
                "id": res["ids"][0][i],
                "title": meta.get("title") or "Unknown",
                "author": meta.get("author") or "",
                "year": meta.get("year"),
                "genres": meta.get("genres") or "",
                "themes": meta.get("themes") or "",
                "document": doc,
                "score": score,
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def debug_collection_info() -> Dict[str, Any]:
    return {"CHROMA_DIR": CHROMA_DIR, "COLLECTION": COLLECTION_NAME, "COUNT": _collection.count()}
