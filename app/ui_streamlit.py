# app/ui_streamlit.py  — UI-only refresh (no feature changes)

# ---- Safe imports for both "module" and "script" run modes ----
# app/ui_streamlit.py
try:
    from .chatbot import recommend_with_tool, record_feedback
    from .speech import transcribe_audio
    from .rag import search_books, debug_collection_info
    from .tools import get_summary_by_title
except Exception:
    from chatbot import recommend_with_tool, record_feedback
    from speech import transcribe_audio
    from rag import search_books, debug_collection_info
    from tools import get_summary_by_title


# (We keep these imports even if not used everywhere; do not remove functionality.)
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration  # noqa: F401

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

load_dotenv(override=True)

# ---------------------- Page & Theme ----------------------
st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")

# Global CSS – tasteful, minimal, professional
st.markdown("""
<style>
/* Layout + typography */
.block-container {max-width: 1080px !important; padding-top: 1.25rem;}
html, body, [class^="css"]  {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial;}
/* Headings */
h1, h2, h3 {letter-spacing: -0.02em;}
/* Card */
.sl-card {
  background: linear-gradient(180deg, rgba(15,23,42,0.70) 0%, rgba(15,23,42,0.50) 100%);
  border: 1px solid rgba(148,163,184,0.20);
  box-shadow: 0 8px 40px rgba(2,6,23,0.25);
  border-radius: 16px; padding: 18px 18px 8px; margin-top: 8px;
}
.sl-pill {
  display:inline-flex; gap:.4rem; align-items:center; padding: .3rem .6rem; border-radius:999px;
  background: rgba(99,102,241,.2); color:#c7d2fe; font-size:.78rem; border:1px solid rgba(99,102,241,.35);
}
/* Buttons */
.stButton>button {
  border-radius: 12px; padding: .6rem 1rem; font-weight: 600;
  border: 1px solid rgba(148,163,184,0.25);
}
.stButton>button:hover { box-shadow: 0 6px 22px rgba(99,102,241,.25); }
/* Text area */
textarea { border-radius: 12px !important; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { background: rgba(148,163,184,.08); border-radius: 12px; padding:.5rem .9rem; }
.stTabs [aria-selected="true"] { background: rgba(99,102,241,.25) !important; color: #fff; }
/* Sidebar */
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);}
section[data-testid="stSidebar"] .stSlider, section[data-testid="stSidebar"] .stToggle {margin-top:.4rem;}
section[data-testid="stSidebar"] .stMarkdown p {color:#cbd5e1;}
/* Live widget */
#sl-live {
  border: 1px solid rgba(148,163,184,.25); border-radius: 14px; padding:14px;
  background: linear-gradient(180deg, rgba(2,6,23,.35) 0%, rgba(2,6,23,.15) 100%);
}
#sl-live button {
  border-radius: 10px; padding: .45rem .8rem; font-weight: 600; border:1px solid rgba(148,163,184,.25);
}
#sl-status { margin-left: 10px; color: #94a3b8; font-weight: 600;}
#sl-out {
  white-space: pre-wrap; max-height: 240px; overflow: auto; padding:10px;
  background: #0b1220; border-radius: 10px; border: 1px dashed rgba(148,163,184,.3); margin-top:10px; color:#e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Hero ----------------------
st.markdown("""
<div class="sl-card" style="padding: 20px;">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <div class="sl-pill">RAG · ChromaDB</div>
    <div class="sl-pill">Tool-calling</div>
    <div class="sl-pill">Realtime STT</div>
    <div class="sl-pill">TTS</div>
    <div class="sl-pill">Image gen</div>
  </div>
  <h1 style="margin:.4rem 0 0;">📚 Smart Librarian</h1>
  <p style="opacity:.9; margin:.3rem 0 0;">
    Recomandări de cărți pe baza intereselor tale, cu rezumat detaliat prin tool-calling.
    Include căutare semantică (RAG), transcriere live prin OpenAI Realtime, TTS și copertă simbolică.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Opțiuni")
    default_k = int(os.getenv("RAG_TOP_K", "5"))
    default_temp = float(os.getenv("CHAT_TEMPERATURE", "0.3"))
    k = st.slider("Top-K (RAG)", min_value=1, max_value=10, value=default_k, step=1, help="Câte pasaje similare să recuperăm din vector store.")
    temperature = st.slider("Temperatură (LLM)", min_value=0.0, max_value=1.0, value=float(default_temp), step=0.05, help="0.0 = foarte factual · 1.0 = mai creativ")
    st.divider()
    tts = st.toggle("🔊 Text-to-Speech pentru răspunsul final", value=False)
    gen_img = st.toggle("🖼️ Generează copertă simbolică", value=False)
    st.caption("**Hint:** TTS și imaginea cresc timpul de răspuns.")

# ---------------------- Tabs ----------------------
tab_text, tab_upload, tab_live_openai = st.tabs([
    "🤖 Recomandare (text)",
    "📁 Voice→Text (fișier)",
    "🎙️ Live (OpenAI Realtime)"
])

# ------------------- TEXT RECOMMEND -------------------
with tab_text:
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)
    st.subheader("Caută și recomandă")
    st.caption("Sugestie: „o carte despre prietenie și magie”, „ceva ca Harry Potter”, „SF militar despre strategie”")
    prompt = st.text_area("Întreabă despre o carte/teme:", placeholder="Ex: Vreau o carte despre prietenie și magie", height=90)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("📖 Recomandă", key="btn_reco_text", use_container_width=True):
            if not prompt.strip():
                st.warning("Te rog scrie o întrebare.")
            else:
                with st.spinner("Găsesc potriviri și pregătesc rezumatul…"):
                    out = recommend_with_tool(prompt.strip(), k=k, temperature=temperature, tts=tts, gen_image=gen_img)
                st.success("Gata!")
                st.markdown(out["text"])
                if out.get("audio"): st.audio(out["audio"])
                if out.get("image"): st.image(out["image"], caption="Copertă simbolică generată")
    with c2:
        st.write("")  # spacer for alignment
        st.write("")
        st.info("🔎 Recomandarea folosește RAG + tool pentru rezumat. Poți ajusta Top-K și temperatura din sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- UPLOAD RECO/SEARCH -------------------
with tab_upload:
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)
    st.subheader("Transcriere fișier audio → Căutare/Rec")
    st.caption("Încarcă **.wav** sau **.mp3**; transcriem local (batch) și folosim textul rezultat.")
    audio_file = st.file_uploader("Fișier audio", type=["wav", "mp3"])
    c1, c2 = st.columns(2)
    if c1.button("🔎 Transcrie & Caută", key="btn_upload_search", use_container_width=True):
        if audio_file is None:
            st.warning("Încarcă un fișier audio.")
        else:
            tmp = Path("assets/tmp"); tmp.mkdir(parents=True, exist_ok=True)
            p = tmp / audio_file.name
            p.write_bytes(audio_file.read())
            with st.spinner("Transcriu…"):
                try:
                    transcript = transcribe_audio(p)
                    st.success("Transcriere finalizată.")
                    st.caption(f"**Text:** {transcript}")
                except Exception as e:
                    st.error(f"Eroare STT: {e}")
                    transcript = ""
            if transcript:
                results = [
                    {
                        "title": c["title"],
                        "snippet": (c["document"][:280] + ("…" if len(c["document"]) > 280 else "")),
                        "score": float(c["score"])
                    } for c in search_books(transcript, k=k)
                ]
                st.subheader("Rezultate căutare (RAG)")
                for i, r in enumerate(results, start=1):
                    with st.expander(f"{i}. {r['title']}  —  sim={r['score']:.3f}"):
                        st.write(r["snippet"])
                        if st.button("📘 Rezumat complet", key=f"u_sum_{i}"):
                            try:
                                st.markdown(get_summary_by_title(r["title"]))
                            except Exception as e:
                                st.error(f"Eroare tool: {e}")
    if c2.button("🤖 Transcrie & Recomandă", key="btn_upload_reco", use_container_width=True):
        if audio_file is None:
            st.warning("Încarcă un fișier audio.")
        else:
            tmp = Path("assets/tmp"); tmp.mkdir(parents=True, exist_ok=True)
            p = tmp / audio_file.name
            p.write_bytes(audio_file.read())
            with st.spinner("Transcriu…"):
                try:
                    transcript = transcribe_audio(p)
                    st.success("Transcriere finalizată.")
                    st.caption(f"**Text:** {transcript}")
                except Exception as e:
                    st.error(f"Eroare STT: {e}")
                    transcript = ""
            if transcript:
                with st.spinner("Generez recomandarea…"):
                    out = recommend_with_tool(transcript, k=k, temperature=temperature, tts=tts, gen_image=gen_img)
                st.markdown(out["text"])
                if out.get("audio"): st.audio(out["audio"])
                if out.get("image"): st.image(out["image"], caption="Copertă simbolică generată")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- LIVE (OpenAI Realtime) -------------------
with tab_live_openai:
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)
    st.subheader("🎙️ Live (OpenAI Realtime)")
    st.caption("Rulează separat: `uvicorn token_server:app --port 5050 --reload`. Apoi Start → vorbește → textul apare live.")
    colA, colB = st.columns(2)
    use_btn = colA.button("⬇️ Folosește ultima transcriere", use_container_width=True)
    clear_btn = colB.button("🧹 Curăță ultima transcriere", use_container_width=True)

    if use_btn:
        import requests
        try:
            r = requests.get("http://localhost:5050/last", timeout=5)
            q = r.json().get("text", "")
            if q:
                st.session_state["last_transcript"] = q
                st.success("Am preluat transcrierea. O poți folosi mai jos.")
            else:
                st.warning("Nu am găsit text încă. Pornește microfonul și vorbește.")
        except Exception as e:
            st.error(f"Eroare la preluarea textului: {e}")

    if clear_btn:
        st.session_state["last_transcript"] = ""

    transcript_box = st.text_area("Transcriere (editabilă):",
                                  value=st.session_state.get("last_transcript", ""),
                                  height=120, key="transcript_area")

    col1, col2 = st.columns(2)
    if col1.button("🔎 Caută (RAG)", use_container_width=True):
        if not transcript_box.strip():
            st.warning("Nu am text. Înregistrează sau editează transcrierea.")
        else:
            results = [
                {
                    "title": c["title"],
                    "snippet": (c["document"][:280] + ("…" if len(c["document"]) > 280 else "")),
                    "score": float(c["score"])
                } for c in search_books(transcript_box, k=int(os.getenv("RAG_TOP_K", "5")))
            ]
            st.subheader("Rezultate căutare (RAG)")
            for i, r in enumerate(results, start=1):
                with st.expander(f"{i}. {r['title']} — sim={r['score']:.3f}"):
                    st.write(r["snippet"])
                    if st.button("📘 Rezumat complet", key=f"live_sum_{i}"):
                        try:
                            st.markdown(get_summary_by_title(r["title"]))
                        except Exception as e:
                            st.error(f"Eroare tool: {e}")

    if col2.button("🤖 Recomandă din transcriere", use_container_width=True):
        if not transcript_box.strip():
            st.warning("Nu am text. Înregistrează sau editează transcrierea.")
        else:
            out = recommend_with_tool(transcript_box,
                                      k=int(os.getenv("RAG_TOP_K", "5")),
                                      temperature=float(os.getenv("CHAT_TEMPERATURE", "0.3")),
                                      tts=False, gen_image=False)
            st.markdown(out["text"])

    # --- Embedded HTML/JS widget (styled) ---
    components.html(f"""
<div id="sl-live">
  <div style="display:flex;gap:8px;align-items:center;">
    <button id="startBtn">Start</button>
    <button id="stopBtn" disabled>Stop</button>
    <span id="sl-status">idle</span>
  </div>
  <pre id="sl-out"></pre>
</div>

<script>
const statusEl = document.getElementById('sl-status');
const out = document.getElementById('sl-out');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

let pc = null;
let dc = null;
let stream = null;

function setStatus(s) {{ statusEl.textContent = s; }}

async function start() {{
  startBtn.disabled = true; stopBtn.disabled = false; setStatus('starting…');

  try {{
    const sess = await fetch('http://localhost:5050/session');
    if (!sess.ok) throw new Error('token server ' + sess.status);
    const {{"client_secret": EPHEMERAL}} = await sess.json();
    if (!EPHEMERAL) throw new Error('no ephemeral token');

    pc = new RTCPeerConnection({{
      iceServers: [{{ urls: 'stun:stun.l.google.com:19302' }}]
    }});
    pc.onconnectionstatechange = () => setStatus('webrtc: ' + pc.connectionState);

    pc.ontrack = (e) => {{
      let audioEl = document.getElementById('oai-audio');
      if (!audioEl) {{
        audioEl = document.createElement('audio');
        audioEl.id = 'oai-audio'; audioEl.autoplay = true;
        document.body.appendChild(audioEl);
      }}
      audioEl.srcObject = e.streams[0];
    }};

    dc = pc.createDataChannel('oai-events');
    dc.onopen = () => {{
      dc.send(JSON.stringify({{
        type: "session.update",
        session: {{
          input_audio_transcription: {{ model: "{os.getenv('REALTIME_TRANSCRIBE_MODEL','gpt-4o-mini-transcribe')}" }}
        }}
      }}));
      setStatus('connected');
    }};
    dc.onmessage = (e) => {{
      try {{
        const msg = JSON.parse(e.data);
        if (msg.type && (msg.type.includes('conversation.item') || msg.type.includes('response'))) {{
          const parts = [];
          if (msg.item && Array.isArray(msg.item.content)) {{
            for (const c of msg.item.content) {{
              if (c.transcript) parts.push(c.transcript);
              if (c.text) parts.push(c.text);
            }}
          }}
          if (msg.delta && msg.delta.transcript) parts.push(msg.delta.transcript);
          const line = parts.join(' ').trim();
          if (line) {{
            out.textContent = (out.textContent + "\\n" + line).trim();
            fetch('http://localhost:5050/push', {{
              method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ text: line, final: (msg.type==='response.done') }})
            }}).catch(()=>{{}});
          }}
        }}
      }} catch(_e) {{}}
    }};

    try {{
      stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    }} catch(e) {{
      throw new Error('mic permission: ' + e.message);
    }}
    stream.getTracks().forEach(t => pc.addTrack(t, stream));

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const model = "{os.getenv('REALTIME_MODEL','gpt-4o-mini-realtime-preview')}";
    const resp = await fetch("https://api.openai.com/v1/realtime?model=" + encodeURIComponent(model), {{
      method: "POST",
      body: offer.sdp,
      headers: {{
        "Authorization": "Bearer " + EPHEMERAL,
        "Content-Type": "application/sdp",
        "OpenAI-Beta": "realtime=v1"
      }},
    }});
    if (!resp.ok) {{
      const t = await resp.text();
      throw new Error('OpenAI: ' + resp.status + ' ' + t);
    }}
    const answer = {{ type: "answer", sdp: await resp.text() }};
    await pc.setRemoteDescription(answer);
  }} catch (err) {{
    setStatus('error: ' + (err?.message || err));
    stopBtn.disabled = true; startBtn.disabled = false;
  }}
}}

function stop() {{
  stopBtn.disabled = true; startBtn.disabled = false; setStatus('stopped');
  try {{ if (dc) dc.close(); }} catch(_){{}}
  try {{ if (pc) pc.close(); }} catch(_){{}}
  if (stream) {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
  pc = null; dc = null;
}}

startBtn.onclick = start;
stopBtn.onclick = stop;
</script>
    """, height=360)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.caption("© Smart Librarian · RAG + Tools + Realtime · built with Streamlit")
