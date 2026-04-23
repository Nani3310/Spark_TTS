import asyncio
import io
import queue
import threading
import time

import numpy as np
import gradio as gr
import soundfile as sf
import torch

# ─────────────────────────────────────────────
# BACKEND DETECTION
# ─────────────────────────────────────────────
HAS_GPU = torch.cuda.is_available()
print(f"[INFO] GPU available: {HAS_GPU}")

if HAS_GPU:
    from cli.SparkTTS import SparkTTS
    _spark = SparkTTS("pretrained_models/Spark-TTS-0.5B", torch.device("cuda:0"))
    print("[INFO] Spark-TTS loaded on GPU ✅")
else:
    print("[INFO] No GPU — using edge-tts ✅")
    try:
        import edge_tts
    except ImportError:
        raise SystemExit("Please run:  pip install edge-tts")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SAMPLE_RATE      = 16000
SENTENCE_ENDS    = {".", "!", "?"}
LEVELS           = ["very_low", "low", "moderate", "high", "very_high"]

# Word-chunk mode: fire TTS every N complete words.
# Smaller = lower latency but choppier prosody.
# Larger = smoother but longer wait.
WORD_CHUNK_SIZE  = 4   # words per TTS call

EDGE_VOICES = {
    "male – India English (Prabhat)"  : "en-IN-PrabhatNeural",
    "female – India English (Neerja)" : "en-IN-NeerjaNeural",
    "male – US English (Guy)"         : "en-US-GuyNeural",
    "female – US English (Jenny)"     : "en-US-JennyNeural",
    "male – UK English (Ryan)"        : "en-GB-RyanNeural",
    "female – UK English (Sonia)"     : "en-GB-SoniaNeural",
}

# ─────────────────────────────────────────────
# EDGE-TTS
# ─────────────────────────────────────────────
async def _edge_bytes(text: str, voice_id: str) -> bytes:
    buf = io.BytesIO()
    communicate = edge_tts.Communicate(text, voice_id)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

def edge_tts_wav(text: str, voice_label: str) -> np.ndarray:
    voice_id = EDGE_VOICES.get(voice_label, "en-IN-PrabhatNeural")
    print(f"[edge-tts] voice_id={voice_id!r}  text={text[:60]!r}")
    mp3_bytes = asyncio.run(_edge_bytes(text, voice_id))
    buf = io.BytesIO(mp3_bytes)
    wav, sr = sf.read(buf)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    if sr != SAMPLE_RATE:
        new_len = int(len(wav) * SAMPLE_RATE / sr)
        wav = np.interp(
            np.linspace(0, len(wav) - 1, new_len),
            np.arange(len(wav)),
            wav,
        ).astype("float32")
    return wav

# ─────────────────────────────────────────────
# SPARK-TTS  (GPU only)
# ─────────────────────────────────────────────
def spark_tts_wav(text, ref_audio, gender, pitch, speed):
    def lvl(v):
        return LEVELS[max(0, min(int(v), 4))] if isinstance(v, (int, float)) else "moderate"
    ref = None
    if isinstance(ref_audio, str):
        ref, _ = sf.read(ref_audio)
    with torch.no_grad():
        wav = _spark.inference(
            text, ref, "",
            gender if gender in ("male", "female") else "male",
            lvl(pitch), lvl(speed),
        )
    return np.array(wav, dtype="float32") if wav is not None else None

# ─────────────────────────────────────────────
# UNIFIED GENERATE
# ─────────────────────────────────────────────
def generate_wav(text, ref_audio, gender, pitch, speed, voice_label):
    if HAS_GPU:
        return spark_tts_wav(text, ref_audio, gender, pitch, speed)
    else:
        return edge_tts_wav(text, voice_label)

# ─────────────────────────────────────────────
# PER-SESSION STATE  (stored in gr.State)
# ─────────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.spoken_upto   = 0        # char index of text already dispatched to TTS
        self.word_buffer   = []       # words accumulated but not yet sent to TTS
        self.audio_chunks  = []       # completed wav arrays in order
        self.result_q      = queue.Queue()
        self.busy_lock     = threading.Lock()
        self.pending_jobs  = 0
        self.stream_mode   = "word"   # "word" or "sentence"

    def reset(self):
        self.spoken_upto  = 0
        self.word_buffer  = []
        self.audio_chunks = []
        self.pending_jobs = 0
        while not self.result_q.empty():
            try:
                self.result_q.get_nowait()
            except queue.Empty:
                break

def make_state():
    return SessionState()

# ─────────────────────────────────────────────
# INFERENCE THREAD  (one per sentence)
# ─────────────────────────────────────────────
def _infer_worker(sentence, ref_audio, gender, pitch, speed, voice_label, state: SessionState):
    try:
        print(f"[TTS→] {sentence!r}")
        wav = generate_wav(sentence, ref_audio, gender, pitch, speed, voice_label)
        if wav is not None and len(wav) > 0:
            state.result_q.put(wav)
            print(f"[TTS✓] {len(wav)/SAMPLE_RATE:.2f}s")
    except Exception:
        import traceback; traceback.print_exc()
    finally:
        with state.busy_lock:
            state.pending_jobs = max(0, state.pending_jobs - 1)

def launch_inference(sentence, ref_audio, gender, pitch, speed, voice_label, state: SessionState):
    with state.busy_lock:
        state.pending_jobs += 1
    t = threading.Thread(
        target=_infer_worker,
        args=(sentence, ref_audio, gender, pitch, speed, voice_label, state),
        daemon=True,
    )
    t.start()

# ─────────────────────────────────────────────
# KEYPRESS HANDLER  — called on every keystroke
# ─────────────────────────────────────────────
def on_type(text, ref_upload, ref_record, gender, pitch, speed, voice_label,
            chunk_size, state: SessionState):
    """
    Word-chunk streaming strategy
    ──────────────────────────────
    We keep a sliding pointer (spoken_upto) into the full text and a
    word_buffer of words seen since the last dispatch.

    Trigger rules (checked in priority order):
      1. A sentence-ending punctuation (. ! ?) flushes the buffer immediately
         regardless of word count — gives natural prosody at sentence breaks.
      2. Every `chunk_size` complete words (followed by a space or punctuation)
         flush the buffer — this is the word-level streaming.

    "Complete word" = a token followed by whitespace or sentence-end punct.
    The last token in the unspoken text is kept in the buffer unless it ends
    with punctuation, because the user may still be typing it.
    """
    if not text:
        state.reset()
        return "Ready — start typing!", state

    # User deleted text behind the spoken pointer → full reset
    if state.spoken_upto > len(text):
        state.reset()

    unspoken = text[state.spoken_upto:]
    if not unspoken:
        return "⌨️  Keep typing…", state

    ref = ref_upload or ref_record
    chunk_size = max(1, int(chunk_size))

    # ── Tokenise the unspoken portion into (word, ends_with_punct) pairs ──
    # We split on whitespace, but preserve trailing punctuation as part of
    # the token so sentence-end detection works correctly.
    import re
    tokens = re.split(r'(\s+)', unspoken)   # keeps whitespace as separate items
    # tokens = [word, ws, word, ws, …]  — filter empties
    tokens = [t for t in tokens if t]

    to_dispatch  = []   # chunks to send to TTS right now
    current_buf  = list(state.word_buffer)
    advance_chars = 0   # chars to move spoken_upto forward

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_ws = bool(re.match(r'^\s+$', tok))

        if is_ws:
            # Whitespace separates words — count the previous word as complete
            advance_chars += len(tok)
            i += 1
            continue

        # It's a word token
        # Check if it ends with sentence punctuation
        ends_sent = tok[-1] in SENTENCE_ENDS

        # Hold this word if the user is still typing it:
        # = it is NOT followed by whitespace AND does NOT end with punctuation
        followed_by_ws = (i + 1 < len(tokens)) and bool(re.match(r'^\s+$', tokens[i + 1]))
        hold = (not followed_by_ws) and (not ends_sent)
        if hold:
            break   # user still mid-word — don't consume yet

        # Consume this token
        advance_chars += len(tok)
        current_buf.append(tok.rstrip())   # strip trailing punct copy for clean join

        # Flush conditions
        flush = ends_sent or (len(current_buf) >= chunk_size)

        if flush:
            phrase = " ".join(current_buf).strip()
            if phrase:
                to_dispatch.append(phrase)
            current_buf = []

        i += 1

    # Save state
    state.spoken_upto += advance_chars
    state.word_buffer  = current_buf

    # Launch TTS for each dispatched chunk
    for phrase in to_dispatch:
        launch_inference(phrase, ref, gender, pitch, speed, voice_label, state)

    # Build status message
    buf_preview = " ".join(state.word_buffer)
    if to_dispatch:
        last = to_dispatch[-1][:50]
        status = f"🔄 Generating: \"{last}\""
        if buf_preview:
            status += f"  |  buffer: \"{buf_preview}\""
    elif buf_preview:
        remaining = chunk_size - len(state.word_buffer)
        status = f"⌨️  buffer [{len(state.word_buffer)}/{chunk_size}]: \"{buf_preview}\"  — {remaining} more word(s) to trigger"
    else:
        status = "⌨️  Keep typing…"

    return status, state

# ─────────────────────────────────────────────
# POLL  — called every second by gr.Timer
# ─────────────────────────────────────────────
def poll(state: SessionState):
    """Drain ALL ready chunks and return concatenated audio so far."""
    got_new = False
    while True:
        try:
            wav = state.result_q.get_nowait()
            if wav is not None and len(wav) > 0:
                state.audio_chunks.append(wav)
                got_new = True
        except queue.Empty:
            break

    if not got_new or not state.audio_chunks:
        return gr.update(), state   # no change → don't flicker the player

    silence = np.zeros(int(0.07 * SAMPLE_RATE), dtype="float32")
    parts = []
    for i, chunk in enumerate(state.audio_chunks):
        if i > 0:
            parts.append(silence)
        parts.append(chunk)
    combined = np.concatenate(parts)
    return (SAMPLE_RATE, combined), state

# ─────────────────────────────────────────────
# CLEAR
# ─────────────────────────────────────────────
def do_clear(state: SessionState):
    state.reset()
    return "", None, "Ready.", state

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #7c6aff;
    --accent2: #ff6a9b;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --success: #4ade80;
    --radius: 12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header */
.header-block {
    text-align: center;
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}

.header-sub {
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 300;
}

.badge {
    display: inline-block;
    background: rgba(124,106,255,0.15);
    border: 1px solid rgba(124,106,255,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    margin-top: 0.75rem;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* Gradio overrides */
label.svelte-1b6s6vi, .label-wrap {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

textarea, input[type="text"] {
    background: #0d0d16 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.2s !important;
    caret-color: var(--accent) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,255,0.15) !important;
    outline: none !important;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    color: var(--success);
    border-radius: 100px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    min-height: 2.2rem;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, var(--accent), #5a4aff) !important;
    border: none !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

button.primary:hover { opacity: 0.85 !important; }

button.secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s !important;
}

button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Slider */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* Radio */
.wrap.svelte-wdgx8n, .wrap {
    gap: 0.5rem !important;
}

/* Audio component */
audio {
    width: 100% !important;
    border-radius: 8px !important;
    background: #0d0d16 !important;
}

/* Dropdown */
select, .dropdown {
    background: #0d0d16 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.25rem 0;
}

/* Waveform animation (purely decorative) */
.waveform-anim {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 20px;
}
.waveform-anim span {
    display: block;
    width: 3px;
    border-radius: 2px;
    background: var(--accent);
    animation: wave 1.2s ease-in-out infinite;
}
.waveform-anim span:nth-child(2) { animation-delay: 0.1s; height: 10px; }
.waveform-anim span:nth-child(3) { animation-delay: 0.2s; height: 16px; }
.waveform-anim span:nth-child(4) { animation-delay: 0.3s; height: 8px;  }
.waveform-anim span:nth-child(5) { animation-delay: 0.4s; height: 14px; }
.waveform-anim span:nth-child(1) { height: 12px; }

@keyframes wave {
    0%, 100% { transform: scaleY(0.5); opacity: 0.5; }
    50%       { transform: scaleY(1.3); opacity: 1;   }
}
"""

def build_ui():
    backend_label = "Spark-TTS · GPU" if HAS_GPU else "edge-tts · CPU"
    accent_color  = "#7c6aff" if HAS_GPU else "#ff6a9b"

    with gr.Blocks(css=CSS, title="Streaming TTS") as demo:

        # ── Header ──────────────────────────────
        gr.HTML(f"""
        <div class="header-block">
            <h1 class="header-title">⚡ Streaming TTS</h1>
            <p class="header-sub">
                Audio fires every <b>N words</b> as you type — no need to finish a sentence.
                Sentence-end punctuation ( <code>.</code> <code>!</code> <code>?</code> ) always flushes immediately.
            </p>
            <span class="badge">{backend_label}</span>
        </div>
        """)

        # ── Session state ────────────────────────
        state = gr.State(make_state)

        # ── Main layout ──────────────────────────
        with gr.Row(equal_height=False):

            # Left column: input + controls
            with gr.Column(scale=5):
                gr.HTML('<div class="panel-label">📝 Input</div>')
                txt = gr.Textbox(
                    label="",
                    placeholder="Hello, my name is Harshinee. I am working on Spark TTS.",
                    lines=7,
                    show_label=False,
                )

                gr.HTML('<hr class="divider">')

                # Voice settings
                if not HAS_GPU:
                    gr.HTML('<div class="panel-label">🎙 Voice</div>')
                    voice_label = gr.Dropdown(
                        choices=list(EDGE_VOICES.keys()),
                        value="male – India English (Prabhat)",
                        label="",
                        show_label=False,
                    )
                else:
                    voice_label = gr.State("male – India English (Prabhat)")

                if HAS_GPU:
                    gr.HTML('<div class="panel-label">🎛 Speaker Controls</div>')
                    with gr.Row():
                        gender = gr.Radio(
                            ["male", "female"], value="male",
                            label="Gender",
                        )
                    with gr.Row():
                        pitch = gr.Slider(0, 4, value=2, step=1, label="Pitch level")
                        speed = gr.Slider(0, 4, value=2, step=1, label="Speed level")

                    gr.HTML('<div class="panel-label" style="margin-top:1rem">🎤 Reference Audio (optional)</div>')
                    ref_upload = gr.Audio(type="filepath", label="Upload reference")
                    ref_record = gr.Audio(type="filepath", label="Record reference", sources=["microphone"])
                else:
                    gender     = gr.State("male")
                    pitch      = gr.State(2)
                    speed      = gr.State(2)
                    ref_upload = gr.State(None)
                    ref_record = gr.State(None)

                gr.HTML('<hr class="divider">')

                gr.HTML('<div class="panel-label">⚡ Streaming Granularity</div>')
                chunk_size = gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="Words per audio chunk",
                    info="1 = fire every word (most responsive, can sound choppy)  |  6–8 = smoother prosody",
                )

                gr.HTML('<hr class="divider">')
                clear_btn = gr.Button("🗑  Clear & Reset", elem_classes=["secondary"])

            # Right column: status + audio out
            with gr.Column(scale=5):
                gr.HTML('<div class="panel-label">📡 Status</div>')
                status = gr.Textbox(
                    label="",
                    value="Ready — start typing!",
                    interactive=False,
                    lines=2,
                    show_label=False,
                )

                gr.HTML("""
                <div class="waveform-anim" style="margin:0.75rem 0">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                """)

                gr.HTML('<div class="panel-label">🔊 Output Audio</div>')
                audio_out = gr.Audio(
                    label="",
                    autoplay=True,
                    show_label=False,
                )

                gr.HTML("""
                <p style="font-size:0.75rem;color:#6b6b80;margin-top:1rem;font-family:'Space Mono',monospace;">
                    💡 Audio accumulates as you type. Lower chunk size = faster first audio.<br>
                    Sentence punctuation always flushes instantly. Hit <b>Clear</b> to restart.
                </p>
                """)

        # ── Event wiring ─────────────────────────
        txt.input(
            fn=on_type,
            inputs=[txt, ref_upload, ref_record, gender, pitch, speed,
                    voice_label, chunk_size, state],
            outputs=[status, state],
        )

        timer = gr.Timer(value=0.8)
        timer.tick(fn=poll, inputs=[state], outputs=[audio_out, state])

        clear_btn.click(
            fn=do_clear,
            inputs=[state],
            outputs=[txt, audio_out, status, state],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=7860, share=False)