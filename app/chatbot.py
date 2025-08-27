import os
import json
import csv
import datetime
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---- Robust imports: works in package *and* script mode ----
try:
    from .rag import search_books
    from .tools import get_summary_by_title
    from .speech import tts_say  # <- TTS helper (pyttsx3)
except Exception:
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from rag import search_books
    from tools import get_summary_by_title
    from speech import tts_say
# ------------------------------------------------------------

load_dotenv(override=True)

CHAT_MODEL = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
TEMP_DEFAULT = float(os.getenv("CHAT_TEMPERATURE", "0.2"))

ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "./assets/covers")).resolve()
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PREFS_PATH = DATA_DIR / "user_prefs.json"
LOG_PATH = DATA_DIR / "log.csv"

client = OpenAI()

BANNED = {"idiot", "stupid", "retard", "disgusting", "fuck", "shit"}  # demo simplu

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Returnează rezumatul complet pentru un titlu exact de carte (dintr-o bază locală).",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False
        }
    }
}]

SYSTEM = (
    "Ești Smart Librarian.\n"
    "Scop: RECOMANDĂ o singură carte DEJA EXISTENTĂ, din CONTEXTUL primit (lista de titluri) sau o carte asemanatoare. "
    "NU inventa titluri, NU scrie proză/ficțiune, NU genera capitole sau pagini. "
    "Dacă utilizatorul spune „ca Harry Potter”, deduci temele și alegi un TITLU DIN CONTEXT. "
    "Primul răspuns TREBUIE să fie NUMAI un apel de tool get_summary_by_title cu titlul ales. "
)

# ---------------------- Preferences ----------------------
def _init_prefs_file():
    if not PREFS_PATH.exists():
        PREFS_PATH.write_text(json.dumps({"liked": [], "disliked": []}, ensure_ascii=False, indent=2), encoding="utf-8")

def load_prefs() -> Dict[str, List[str]]:
    _init_prefs_file()
    try:
        return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"liked": [], "disliked": []}

def save_prefs(p: Dict[str, List[str]]):
    PREFS_PATH.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

def record_feedback(title: str, liked: bool):
    p = load_prefs()
    tl = title.strip()
    if liked:
        if tl not in p["liked"]:
            p["liked"].append(tl)
        if tl in p["disliked"]:
            p["disliked"].remove(tl)
    else:
        if tl not in p["disliked"]:
            p["disliked"].append(tl)
        if tl in p["liked"]:
            p["liked"].remove(tl)
    save_prefs(p)

# ---------------------- Logging -------------------------
def log_interaction(query: str, picked_title: str | None, picked_score: float | None):
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts_utc", "query", "picked_title", "picked_score"])
        w.writerow([
            datetime.datetime.utcnow().isoformat(),
            query,
            picked_title or "",
            f"{picked_score:.4f}" if isinstance(picked_score, (int, float)) else ""
        ])

# -------------------- Helpers ----------------------
def is_inappropriate(text: str) -> bool:
    t = text.lower()
    return any(bad in t for bad in BANNED)

def _apply_personalization(context: List[Dict]) -> None:
    prefs = load_prefs()
    liked = set([t.lower() for t in prefs.get("liked", [])])
    disliked = set([t.lower() for t in prefs.get("disliked", [])])
    for c in context:
        t = c["title"].lower()
        if t in liked:    c["score"] += 0.05
        if t in disliked: c["score"] -= 0.05
    context.sort(key=lambda x: x["score"], reverse=True)

def _looks_like_generation_request(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "scrie", "scrie-mi", "creează", "creeaza", "compune",
        "story", "fanfic", "roman", "capitol", "poveste", "invent",
        "write a", "create a", "generate a"
    ]
    return any(t in ql for t in triggers)

# -------------------- Public API ------------------------
def recommend_with_tool(
    user_query: str,
    k: int | None = None,
    temperature: float | None = None,
    tts: bool = False,
    gen_image: bool = False
) -> Dict:
    if is_inappropriate(user_query):
        return {"text": "Prefer să păstrez conversația respectuoasă. Te rog reformulează fără cuvinte ofensatoare.",
                "audio": None, "image": None, "picked_title": None, "picked_score": None}

    # 1) RAG
    candidates = search_books(user_query, k=k)
    context = [
        {
            "title": c["title"],
            "snippet": (c["document"][:280] + ("…" if len(c["document"]) > 280 else "")),
            "score": round(float(c["score"]), 4)
        }
        for c in candidates
    ]
    _apply_personalization(context)
    top_ctx = context[:3]

    # 2) Mesaje: întărim intenția de RECOMANDARE
    intent_guard = (
        "User intent: RECOMMENDATION ONLY. "
        "If the user asked to 'write a story', re-interpret as 'recommend an existing book'. "
        "Never write story text."
    )
    if _looks_like_generation_request(user_query):
        user_query = f"{user_query}\n(Notă: tratează ca cerere de RECOMANDARE, nu de generare de text.)"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": intent_guard},
        {"role": "assistant", "content": "CONTEXT CANDIDATE: " + json.dumps(context, ensure_ascii=False)}
    ]

    temp = float(temperature if temperature is not None else TEMP_DEFAULT)

    # 3) PRIMUL PAS: forțează DOAR apel de tool cu titlul ales
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice={"type":"function", "function":{"name":"get_summary_by_title"}},
        temperature=temp,
        max_tokens=200
    )
    msg = first.choices[0].message

    picked_title = None
    picked_score = None
    text = ""

    # 4) Executăm tool-ul (obligatoriu)
    summary = ""
    if msg.tool_calls:
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type":"function",
                 "function":{"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })
        for tc in msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments)
                picked_title = args.get("title")
                if picked_title:
                    for c in context:
                        if c["title"].lower() == picked_title.lower():
                            picked_score = c["score"]
                            break
                try:
                    summary = get_summary_by_title(picked_title)
                except Exception as e:
                    summary = f"Eroare tool: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary
                })

        # 5) AL DOILEA PAS: cerem DOAR 2–3 motive (bullets), fără ficțiune
        reasons_system = (
            "Generează exclusiv o listă de 2-3 bullet-uri scurte cu MOTIVE pentru care titlul ales se potrivește "
            "cererii utilizatorului. Nu scrie poveste/ficțiune. Nu inventa detalii. "
            "Fără titlu înapoi, doar bullet-urile."
        )
        messages2 = [
            {"role": "system", "content": reasons_system},
            {"role": "user", "content": f"Cerere utilizator: {user_query}"},
            {"role": "assistant", "content": "CONTEXT CANDIDATE: " + json.dumps(context, ensure_ascii=False)},
            {"role": "assistant", "content": f"Titlul ales: {picked_title}"},
            {"role": "assistant", "content": f"Rezumat (din tool) pentru context, nu de rescris: {summary[:800]}"}
        ]
        reasons = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages2,
            temperature=0.1,
            max_tokens=200
        ).choices[0].message.content or ""

        # 6) Compunem răspunsul final
        text = f"**Recomandare:** {picked_title}\n\n**De ce:**\n{reasons}\n\n**Rezumat detaliat:**\n{summary}"

    else:
        text = "Nu am reușit să aleg un titlu din context. Încearcă să reformulezi întrebarea."

    # Citări
    if top_ctx:
        cite_lines = "\n".join([f"- **{c['title']}** (sim={c['score']:.3f}) – {c['snippet']}" for c in top_ctx])
        text = f"{text}\n\n**Context folosit (RAG):**\n{cite_lines}"

    # Log
    log_interaction(user_query, picked_title, picked_score)

    # -------- TTS (toggle) --------
    audio_path = None
    if tts and text:
        try:
            audio_path = tts_say(text, ASSETS_DIR)
        except Exception:
            audio_path = None

    # -------- Image generation (toggle) --------
    image_path = None
    if gen_image:
        try:
            prompt = f"Minimalist symbolic book cover that fits the themes of '{picked_title}'."
            img = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024", n=1)
            import base64, io
            from PIL import Image
            raw = base64.b64decode(img.data[0].b64_json)
            image_path = ASSETS_DIR / "cover.png"
            Image.open(io.BytesIO(raw)).save(str(image_path))
        except Exception:
            image_path = None

    return {
        "text": text,
        "audio": str(audio_path) if audio_path else None,
        "image": str(image_path) if image_path else None,
        "picked_title": picked_title,
        "picked_score": picked_score
    }
