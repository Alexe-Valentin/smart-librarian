# app/tools.py
from __future__ import annotations
import os, json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

DATA_JSON = os.getenv("DATA_JSON", "./data/book_summaries.json")

def _norm(s: str) -> str:
    """Lowercase, remove diacritics, collapse non-alnum → match titles robustly."""
    s = unicodedata.normalize("NFKD", s).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def _load_data() -> Any:
    p = Path(DATA_JSON)
    if not p.exists():
        raise FileNotFoundError(f"DATA_JSON not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _summary_from_dict(data: Dict[str, Any], title: str) -> Optional[str]:
    # exact
    if title in data:
        val = data[title]
        return val if isinstance(val, str) else (val.get("summary") if isinstance(val, dict) else None)
    # case/normalize
    nt = _norm(title)
    for k, v in data.items():
        if _norm(k) == nt:
            return v if isinstance(v, str) else (v.get("summary") if isinstance(v, dict) else None)
    return None

def _summary_from_list(data: List[Dict[str, Any]], title: str) -> Optional[str]:
    nt = _norm(title)
    # exact normalized
    for rec in data:
        if _norm(str(rec.get("title", ""))) == nt:
            return str(rec.get("summary") or "")
    # fallback: contains
    for rec in data:
        t = _norm(str(rec.get("title", "")))
        if nt in t or t in nt:
            return str(rec.get("summary") or "")
    return None

def get_summary_by_title(title: str) -> str:
    """
    Returnează rezumatul complet pentru titlul exact (robust la diacritice/punctuație).
    Suportă:
      - dict: { "Title": "summary", ... }
      - list: [ { "title": "...", "summary": "...", "genres": [...], "themes": [...] }, ... ]
    """
    if not title or not str(title).strip():
        return "Titlu invalid."
    data = _load_data()
    summary: Optional[str] = None

    if isinstance(data, dict):
        summary = _summary_from_dict(data, title)
    elif isinstance(data, list):
        summary = _summary_from_list(data, title)
    else:
        return "Formatul DATA_JSON nu este suportat (nu e dict sau list)."

    if summary:
        return summary.strip()

    return f"Nu am găsit rezumat pentru „{title}” în baza locală."
