import os
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment.")

log = logging.getLogger("realtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_text: str = ""

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/session")
async def create_session():
    """
    Mint an ephemeral Realtime session token.
    If your network blocks api.openai.com, we'll return a 502 with details.
    """
    payload = {
        "model": REALTIME_MODEL,
        "input_audio_transcription": {"model": TRANSCRIBE_MODEL},
    }
    log.info("POST /v1/realtime/sessions starting")

    # trust_env=True makes httpx honor HTTPS_PROXY/HTTP_PROXY from your env
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=True) as client:
            r = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1",
                },
                json=payload,
            )
            log.info("OpenAI response status: %s", r.status_code)
            r.raise_for_status()
            data = r.json()
            token: Optional[str] = data.get("client_secret", {}).get("value")
            if not token:
                return JSONResponse({"error": "No client_secret.value in response"}, status_code=502)
            return {"client_secret": token}

    except httpx.HTTPError as e:
        log.exception("Error calling OpenAI Realtime")
        # Bubble up a readable error to the browser/your test call
        return JSONResponse({"error": str(e)}, status_code=502)

@app.post("/push")
async def push_transcript(req: Request):
    global _last_text
    body = await req.json()
    txt = (body.get("text") or "").strip()
    final = bool(body.get("final"))
    if txt and final:
        _last_text = (_last_text + " " + txt).strip() if _last_text else txt
    return {"ok": True, "current": _last_text}

@app.get("/last")
async def get_last():
    return {"text": _last_text}
