"""
AI Video Agent - Streamlit UI

Put this file next to main.py (project root), then run:
    uv pip install streamlit
    uv run streamlit run app.py
"""
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE importing anything from core/ or utils/
load_dotenv(Path(__file__).parent / ".env", override=True)

import html
import os
import re
import tempfile
import time
import traceback

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

st.set_page_config(
    page_title="AI Video Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling + animations
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
footer { visibility: hidden; }

/* ---------- keyframes ---------- */
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes fadeInUp { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:translateY(0)} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(99,102,241,.55)} 70%{box-shadow:0 0 0 12px rgba(99,102,241,0)} 100%{box-shadow:0 0 0 0 rgba(99,102,241,0)} }
@keyframes spin { to{transform:rotate(360deg)} }
@keyframes bounce { 0%,80%,100%{transform:scale(.6); opacity:.5} 40%{transform:scale(1); opacity:1} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

/* ---------- hero ---------- */
.hero {
  position:relative; overflow:hidden; padding:2.2rem 2.2rem 1.8rem; border-radius:24px; color:#fff;
  background:linear-gradient(120deg,#6366f1,#8b5cf6,#ec4899,#f59e0b); background-size:300% 300%;
  animation:gradientShift 10s ease infinite, fadeInUp .7s ease both;
  box-shadow:0 24px 60px -24px rgba(99,102,241,.7); margin-bottom:1.2rem;
}
.hero::before, .hero::after { content:""; position:absolute; border-radius:50%; background:rgba(255,255,255,.18); }
.hero::before { width:220px; height:220px; right:-50px; top:-70px; animation:float 7s ease-in-out infinite; }
.hero::after  { width:120px; height:120px; right:140px; bottom:-60px; animation:float 5s ease-in-out infinite reverse; }
.hero-title { font-size:2.5rem; font-weight:800; letter-spacing:-.02em; line-height:1.15; position:relative; z-index:1; }
.hero-sub { margin-top:.5rem; font-size:1.05rem; opacity:.95; max-width:640px; position:relative; z-index:1; }
.chips { margin-top:1rem; display:flex; flex-wrap:wrap; gap:.5rem; position:relative; z-index:1; }
.chips span { padding:.3rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600;
  background:rgba(255,255,255,.2); backdrop-filter:blur(6px); border:1px solid rgba(255,255,255,.3); }

/* ---------- feature cards (empty state) ---------- */
.feature { padding:1.4rem; border-radius:18px; border:1px solid rgba(128,128,128,.22);
  background:rgba(128,128,128,.06); animation:fadeInUp .7s ease both; height:100%;
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
.feature:hover { transform:translateY(-6px); border-color:#8b5cf6; box-shadow:0 18px 36px -18px rgba(139,92,246,.6); }
.f-icon { font-size:2rem; display:inline-block; animation:float 4s ease-in-out infinite; }
.f-title { font-weight:700; font-size:1.05rem; margin-top:.4rem; }
.f-desc { opacity:.75; font-size:.92rem; margin-top:.2rem; }

/* ---------- stepper ---------- */
.stepper { display:flex; gap:.6rem; flex-wrap:wrap; margin:.8rem 0 .4rem; }
.step { flex:1 1 150px; padding:.7rem .9rem; border-radius:14px; display:flex; align-items:center; gap:.6rem;
  border:1px solid rgba(128,128,128,.25); background:rgba(128,128,128,.07); font-size:.9rem;
  transition:all .35s ease; }
.step .dot { width:26px; height:26px; border-radius:50%; display:grid; place-items:center; flex:none;
  font-size:.78rem; font-weight:700; background:rgba(128,128,128,.28); }
.step.active { border-color:#6366f1; background:rgba(99,102,241,.14); animation:pulse 1.6s infinite; }
.step.active .dot { background:#6366f1; color:#fff; }
.step.done { border-color:rgba(16,185,129,.55); background:rgba(16,185,129,.10); }
.step.done .dot { background:#10b981; color:#fff; }
.spinner { width:14px; height:14px; border:2px solid rgba(255,255,255,.45); border-top-color:#fff;
  border-radius:50%; animation:spin .8s linear infinite; display:inline-block; }
.note { margin:.2rem 0 .8rem; padding:.7rem 1rem; border-radius:12px; font-weight:500;
  background:linear-gradient(90deg, rgba(99,102,241,.10), rgba(236,72,153,.10), rgba(99,102,241,.10));
  background-size:200% 100%; animation:shimmer 3s linear infinite; }

/* ---------- results ---------- */
.result-title { font-size:1.9rem; font-weight:800; letter-spacing:-.01em; margin:.2rem 0 1rem; animation:fadeInUp .6s ease both;
  background:linear-gradient(90deg,#6366f1,#ec4899,#f59e0b); background-size:200% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:fadeInUp .6s ease both, gradientShift 8s ease infinite; }
.stat { text-align:center; padding:1rem .6rem; border-radius:16px; border:1px solid rgba(128,128,128,.22);
  background:linear-gradient(145deg, rgba(99,102,241,.13), rgba(236,72,153,.08)); animation:fadeInUp .6s ease both;
  transition:transform .25s ease, box-shadow .25s ease; }
.stat:hover { transform:translateY(-5px); box-shadow:0 16px 32px -16px rgba(99,102,241,.6); }
.stat .num { font-size:1.8rem; font-weight:800; background:linear-gradient(90deg,#6366f1,#ec4899);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.stat .lbl { font-size:.74rem; opacity:.7; text-transform:uppercase; letter-spacing:.08em; margin-top:.15rem; }
[class*="st-key-card_"] { animation:fadeInUp .5s ease both; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { border-radius:16px; animation:fadeInUp .4s ease both; }
.typing { display:inline-flex; gap:5px; padding:.5rem 0; }
.typing span { width:8px; height:8px; border-radius:50%; background:#8b5cf6; animation:bounce 1.2s infinite ease-in-out; }
.typing span:nth-child(2){ animation-delay:.15s } .typing span:nth-child(3){ animation-delay:.3s }

/* ---------- buttons / tabs ---------- */
.stButton > button, .stDownloadButton > button { border-radius:12px; font-weight:600;
  transition:transform .2s ease, box-shadow .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 22px -10px rgba(99,102,241,.75); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899); background-size:200% 200%;
  border:none; color:#fff; animation:gradientShift 6s ease infinite; }
[data-testid="stSidebar"] .stButton > button { width:100%; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; transition:background .2s ease; }
.stTabs [data-baseweb="tab"]:hover { background:rgba(99,102,241,.12); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
STEPS = [
    ("🎧", "Process audio"),
    ("📝", "Transcribe"),
    ("🏷️", "Title"),
    ("📋", "Summary"),
    ("✅", "Action items"),
    ("🔑", "Decisions"),
    ("❓", "Open questions"),
    ("🧠", "Chat index"),
]

SUGGESTIONS = [
    "What were the main topics discussed?",
    "Who is responsible for what?",
    "What deadlines were mentioned?",
]

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean(text) -> str:
    """Normalize LLM output to a string and strip any <think>...</think> block."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return THINK_RE.sub("", text).strip()


def count_items(text: str) -> int:
    return len(re.findall(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", text or "", re.MULTILINE))


def render_stepper(box, active: int, finished: bool = False) -> None:
    parts = []
    for i, (icon, label) in enumerate(STEPS):
        if finished or i < active:
            cls, dot = "done", "✓"
        elif i == active:
            cls, dot = "active", '<span class="spinner"></span>'
        else:
            cls, dot = "", str(i + 1)
        parts.append(
            f'<div class="step {cls}"><div class="dot">{dot}</div><div>{icon} {label}</div></div>'
        )
    box.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def cooldown(seconds: int, box) -> None:
    """Animated countdown to stay under Groq's free-tier tokens-per-minute limit."""
    if seconds <= 0:
        return
    with box.container():
        msg = st.empty()
        bar = st.progress(0.0)
        for i in range(seconds):
            msg.markdown(f"⏳ Cooling down for the Groq rate limit… **{seconds - i}s**")
            bar.progress((i + 1) / seconds)
            time.sleep(1)
    box.empty()


def typewriter(box, text: str) -> None:
    words = text.split(" ")
    delay = min(0.03, 2.5 / max(len(words), 1))
    shown = ""
    for i, w in enumerate(words):
        shown += ("" if i == 0 else " ") + w
        box.markdown(shown + " ▌")
        time.sleep(delay)
    box.markdown(shown)


def save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "403" in msg or "forbidden" in low or "sign in to confirm" in low:
        return (
            "YouTube blocked the download from this server (cloud servers are usually blocked). "
            "Use the 'Upload file' option instead, or run the app on your own PC for YouTube links."
        )
    if "429" in msg or "rate limit" in low or "rate_limit" in low:
        return "Groq rate limit hit. Increase the pause in the sidebar (or wait a minute) and try again."
    if "401" in msg or "invalid api key" in low:
        return "The API key was rejected. Check GROQ_API_KEY in your .env file."
    if "ffmpeg" in low or "ffprobe" in low:
        return "ffmpeg wasn't found. Install it with `winget install Gyan.FFmpeg`, then restart VS Code."
    if "no module named" in low:
        return f"A Python package is missing: {msg}. Install it with `uv pip install <package>`."
    return f"Something went wrong: {msg}"


def build_report(res: dict) -> str:
    return (
        f"# {res['title']}\n\n"
        f"## Summary\n{res['summary']}\n\n"
        f"## Action items\n{res['action_items']}\n\n"
        f"## Key decisions\n{res['key_decisions']}\n\n"
        f"## Open questions\n{res['open_questions']}\n"
    )


def reset_session() -> None:
    st.session_state.result = None
    st.session_state.chat = []
    st.session_state.view = "📊 Insights"


# ----------------------------------------------------------------------------
# Pipeline (runs step by step so the UI can animate progress)
# ----------------------------------------------------------------------------
def run_pipeline_ui(source: str, language: str, pause: int) -> dict:
    stepper = st.empty()
    note = st.empty()
    pause_box = st.empty()

    def go(i: int, text: str) -> None:
        render_stepper(stepper, i)
        note.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)

    go(0, "Fetching and splitting the audio…")
    chunks = process_input(source)
    try:
        n = len(chunks)
    except TypeError:
        n = "?"

    go(1, f"Transcribing {n} audio chunk(s) - this is the slow part…")
    transcript = transcribe_all(chunks, language=language)

    go(2, "Writing a title…")
    title = clean(generate_title(transcript)).strip('"\' ')
    cooldown(pause, pause_box)

    go(3, "Summarizing the transcript…")
    summary = clean(summarize(transcript))
    cooldown(pause, pause_box)

    go(4, "Extracting action items…")
    action_items = clean(extract_action_items(transcript))
    cooldown(pause, pause_box)

    go(5, "Finding key decisions…")
    key_decisions = clean(extract_key_decisions(transcript))
    cooldown(pause, pause_box)

    go(6, "Collecting open questions…")
    open_questions = clean(extract_questions(transcript))

    go(7, "Building the chat index (embeddings)…")
    rag_chain = build_rag_chain(transcript)

    render_stepper(stepper, len(STEPS), finished=True)
    note.markdown('<div class="note">✨ All done!</div>', unsafe_allow_html=True)

    return {
        "title": title or "Untitled meeting",
        "transcript": str(transcript),
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------
def hero() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">🎬 AI Video Agent</div>'
        '<div class="hero-sub">Turn any video or meeting into a transcript, summary, action items '
        "and a chat-ready knowledge base.</div>"
        '<div class="chips"><span>YouTube</span><span>Local files</span>'
        "<span>English · Hinglish</span><span>RAG chat</span></div></div>",
        unsafe_allow_html=True,
    )


def features() -> None:
    items = [
        ("🔗", "Drop in a source", "Paste a YouTube link or upload an audio / video file."),
        ("⚙️", "Watch the agent work", "Transcription, summary and extraction run step by step."),
        ("💬", "Chat with it", "Ask questions. Answers come from the transcript via RAG."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(items):
        cols[i].markdown(
            f'<div class="feature" style="animation-delay:{i * 0.12:.2f}s">'
            f'<div class="f-icon">{icon}</div><div class="f-title">{title}</div>'
            f'<div class="f-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )


def insights_view(res: dict) -> None:
    st.markdown(f'<div class="result-title">📌 {html.escape(res["title"])}</div>', unsafe_allow_html=True)

    words = len(res["transcript"].split())
    stats = [
        (f"{words:,}", "Words transcribed"),
        (f"{max(1, round(words / 150))} min", "Est. talk time"),
        (str(count_items(res["action_items"]) or "-"), "Action items"),
        (str(count_items(res["open_questions"]) or "-"), "Open questions"),
    ]
    cols = st.columns(4)
    for i, (num, lbl) in enumerate(stats):
        cols[i].markdown(
            f'<div class="stat" style="animation-delay:{i * 0.1:.1f}s">'
            f'<div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    tabs = st.tabs(["📋 Summary", "✅ Action items", "🔑 Decisions", "❓ Open questions", "📝 Transcript"])
    bodies = [
        ("summary", res["summary"]),
        ("actions", res["action_items"]),
        ("decisions", res["key_decisions"]),
        ("questions", res["open_questions"]),
    ]
    for tab, (name, body) in zip(tabs[:4], bodies):
        with tab:
            with st.container(border=True, key=f"card_{name}"):
                st.markdown(body or "_Nothing found._")
    with tabs[4]:
        with st.container(border=True, key="card_transcript"):
            st.text_area("Transcript", res["transcript"], height=420, label_visibility="collapsed")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Report (.md)", build_report(res), file_name="meeting_report.md", mime="text/markdown")
    d2.download_button("⬇️ Transcript (.txt)", res["transcript"], file_name="transcript.txt", mime="text/plain")


def chat_view(res: dict) -> None:
    st.markdown("#### 💬 Chat with your video")

    if not st.session_state.chat:
        with st.chat_message("assistant"):
            st.markdown("Hi! I've read the whole transcript. Ask me anything about it.")
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            if col.button(text, key=f"sugg_{text}"):
                st.session_state.pending_q = text
                st.rerun()

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about the video…")
    if not question:
        question = st.session_state.pop("pending_q", None)

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            box = st.empty()
            box.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            try:
                answer = clean(ask_question(res["rag_chain"], question)) or "I couldn't find that in the transcript."
            except Exception as e:  # noqa: BLE001
                answer = f"⚠️ {friendly_error(e)}"
            typewriter(box, answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
for key, default in {"result": None, "chat": [], "just_finished": False, "view": "📊 Insights"}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    language = st.selectbox("Spoken language", ["english", "hinglish"])
    pause = st.slider(
        "Pause between AI steps (sec)", 0, 60, 15,
        help="Groq's free tier limits output tokens per minute. Set to 0 if you use a model with higher limits.",
    )
    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY loaded", icon="🔑")
    else:
        st.error("GROQ_API_KEY not found in .env", icon="⚠️")
    st.button("🔄 New video", on_click=reset_session)
    st.caption("Built with Streamlit · LangChain · Groq")

hero()

if st.session_state.result is None:
    panel = st.empty()
    with panel.container(border=True):
        on_cloud = Path("/mount/src").exists()  # True on Streamlit Community Cloud
        options = ["📁 Upload file", "🔗 YouTube link"] if on_cloud else ["🔗 YouTube link", "📁 Upload file"]
        mode = st.radio("Source", options, horizontal=True, label_visibility="collapsed")
        url, uploaded = "", None
        if mode.startswith("🔗"):
            if on_cloud:
                st.warning(
                    "YouTube usually blocks downloads from cloud servers, so this may fail. "
                    "Uploading an audio or video file is the reliable option here.",
                    icon="⚠️",
                )
            url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", label_visibility="collapsed")
        else:
            uploaded = st.file_uploader(
                "Audio or video file",
                type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "mkv", "mov", "webm"],
                label_visibility="collapsed",
            )
        ready = bool(url.strip()) or uploaded is not None
        clicked = st.button("🚀 Analyze", type="primary", disabled=not ready)

    if clicked and ready:
        panel.empty()
        try:
            source = url.strip() if url.strip() else save_upload(uploaded)
            st.session_state.result = run_pipeline_ui(source, language, pause)
            st.session_state.chat = []
            st.session_state.just_finished = True
        except Exception as e:  # noqa: BLE001
            st.error(friendly_error(e), icon="🚨")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            st.button("↩️ Try again")
        else:
            st.rerun()
    elif not clicked:
        st.write("")
        features()
else:
    if st.session_state.just_finished:
        st.session_state.just_finished = False
        st.balloons()
        st.toast("Analysis complete!", icon="✅")

    view = st.radio("View", ["📊 Insights", "💬 Chat"], horizontal=True, key="view", label_visibility="collapsed")
    if view.startswith("📊"):
        insights_view(st.session_state.result)
    else:
        chat_view(st.session_state.result)