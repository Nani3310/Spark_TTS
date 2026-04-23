import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import platform
import librosa
import numpy as np
import torch
from datetime import datetime
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# ── WATERMARK ─────────────────────────────────────────────────────────────────
from watermark import embed_watermarks, detect_public, detect_private

# ── RL AGENT ──────────────────────────────────────────────────────────────────
from rl_agent import TTSRLAgent
agent = TTSRLAgent()
# ─────────────────────────────────────────────────────────────────────────────

# Store last generation info so the rating callback knows what to update
_last_gen = {"text": "", "gender": "male", "pitch": 3, "speed": 3}


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    logging.info(f"Loading model from: {model_dir}")
    if platform.system() == "Darwin":
        device = torch.device(f"mps:{device}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")
    model = SparkTTS(model_dir, device)
    return model


def load_wav_16k_mono(path: str, target_sr: int = 16000):
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    if len(audio) == 0:
        return np.zeros(target_sr, dtype=np.float32)
    return audio.astype(np.float32)


WAVLM_EXTRACTOR = None
WAVLM_MODEL     = None

def get_wavlm_model():
    global WAVLM_EXTRACTOR, WAVLM_MODEL
    if WAVLM_EXTRACTOR is None:
        WAVLM_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        WAVLM_MODEL     = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
        WAVLM_MODEL.eval()
    return WAVLM_EXTRACTOR, WAVLM_MODEL

def wavlm_cosine_similarity(ref_path, test_path, sr=16000):
    extractor, wmodel = get_wavlm_model()
    ar = load_wav_16k_mono(ref_path, sr)
    at = load_wav_16k_mono(test_path, sr)
    ir = extractor(ar, sampling_rate=sr, return_tensors="pt", padding=True)
    it = extractor(at, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        er = wmodel(**ir).embeddings
        et = wmodel(**it).embeddings
    er = torch.nn.functional.normalize(er, dim=-1)
    et = torch.nn.functional.normalize(et, dim=-1)
    return torch.cosine_similarity(er, et, dim=-1).item()


def run_tts(
    text, model,
    prompt_text=None, prompt_speech=None,
    gender=None, pitch=None, speed=None,
    save_dir="example/results",
):
    if prompt_text is not None and len(prompt_text.strip()) == 0:
        prompt_text = None

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    with torch.no_grad():
        wav = model.inference(text, prompt_speech, prompt_text, gender, pitch, speed)

    # ── WATERMARK ─────────────────────────────────────────────────────────────
    try:
        wav = embed_watermarks(np.array(wav, dtype="float32"), sample_rate=16000)
        logging.info("Watermarks embedded ✅")
    except Exception as e:
        logging.warning(f"Watermarking failed: {e}")
        wav = np.array(wav, dtype="float32")
    # ─────────────────────────────────────────────────────────────────────────

    sf.write(save_path, wav, samplerate=16000)
    return save_path


# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────
def verify_audio(audio_file):
    if audio_file is None:
        return "Upload an audio file first.", ""
    try:
        wav, sr = sf.read(audio_file)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
    except Exception as e:
        return f"❌ Cannot read file: {e}", ""

    pub = detect_public(wav,  sample_rate=sr)
    prv = detect_private(wav, sample_rate=sr)

    pub_txt = (
        f"PUBLIC WATERMARK  (AudioSeal — anyone can check)\n"
        f"  Result     : {pub['message']}\n"
        f"  Confidence : {pub['confidence']}"
    )
    prv_txt = (
        f"PRIVATE WATERMARK  (Spread-Spectrum — only this system)\n"
        f"  Result     : {prv['message']}\n"
        f"  Score      : {prv['score']}  (threshold: {prv['threshold']})"
    )
    return pub_txt, prv_txt


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def build_ui(model_dir, device=0):
    model = initialize_model(model_dir, device=device)

    # ── Voice Clone ──────────────────────────────────────────────────────────
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        prompt_speech     = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = prompt_text if prompt_text and len(prompt_text.strip()) > 1 else None

        audio_output_path = run_tts(
            text, model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
        )

        if prompt_speech and os.path.exists(prompt_speech):
            try:
                sim     = wavlm_cosine_similarity(prompt_speech, audio_output_path)
                metrics = f"{sim:.4f} (WavLM Cosine Similarity)"
            except Exception as e:
                metrics = f"Error computing similarity: {e}"
        else:
            metrics = "N/A (No reference audio)"

        return audio_output_path, metrics

    # ── Voice Creation with RL ───────────────────────────────────────────────
    def voice_creation_rl(text, use_rl, gender, pitch, speed):
        """
        If use_rl=True  → agent suggests parameters (ignores sliders)
        If use_rl=False → use slider values directly
        """
        global _last_gen

        if use_rl:
            # Agent suggests best known params for this text
            sug_gender, sug_pitch, sug_speed = agent.suggest(text)
            actual_gender = sug_gender
            actual_pitch  = sug_pitch
            actual_speed  = sug_speed
            rl_note = (
                f"🤖 RL Agent suggested: gender={sug_gender} | "
                f"pitch={sug_pitch} | speed={sug_speed}"
            )
        else:
            actual_gender = gender
            actual_pitch  = int(pitch)
            actual_speed  = int(speed)
            rl_note = "🎛️ Manual mode — using your slider values"

        # Save for the rating callback
        _last_gen = {
            "text"  : text,
            "gender": actual_gender,
            "pitch" : actual_pitch,
            "speed" : actual_speed,
        }

        pitch_val = LEVELS_MAP_UI[actual_pitch]
        speed_val = LEVELS_MAP_UI[actual_speed]

        audio_output_path = run_tts(
            text, model,
            gender=actual_gender,
            pitch=pitch_val,
            speed=speed_val,
        )

        return audio_output_path, rl_note, "⭐ Rate this output to train the RL agent →"

    def submit_rating(stars: int):
        """Called when user clicks a star rating button."""
        if not _last_gen["text"]:
            return "No generation to rate yet."

        agent.update(
            text   = _last_gen["text"],
            gender = _last_gen["gender"],
            pitch  = _last_gen["pitch"],
            speed  = _last_gen["speed"],
            stars  = stars,
        )
        short_text = _last_gen["text"][:30]
        return f"✅ Thanks! Rated {stars}⭐ — agent updated for: '{short_text}…'"

    def get_rl_stats():
        return agent.get_stats()

    # ─────────────────────────────────────────────────────────────────────────
    with gr.Blocks() as demo:
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        gr.Markdown(
            "> 🔐 Every clip is **watermarked** (public + private).  "
            "🤖 The **RL agent** learns your preferred voice over time."
        )

        with gr.Tabs():

            # ── Tab 1: Voice Clone ────────────────────────────────────────────
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio\n"
                    "**Tip:** Fill in the transcript below for higher speaker similarity."
                )
                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        type="filepath",
                        label="Upload reference audio (≥5s recommended)",
                    )
                    prompt_wav_record = gr.Audio(
                        type="filepath",
                        label="Or record reference audio",
                    )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to synthesise", lines=3,
                        placeholder="Enter the text you want spoken here",
                    )
                    prompt_text_input = gr.Textbox(
                        label="📝 Transcript of reference audio (recommended)",
                        lines=3,
                        placeholder="Type exactly what is said in the reference audio.",
                    )

                audio_output   = gr.Audio(label="Generated Audio (watermarked 🔐)", autoplay=True)
                similarity_out = gr.Textbox(label="Speaker Similarity", interactive=False)
                gr.Button("Generate").click(
                    voice_clone,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[audio_output, similarity_out],
                )

            # ── Tab 2: Voice Creation + RL ────────────────────────────────────
            with gr.TabItem("Voice Creation  🤖 RL"):
                gr.Markdown("""
                ### RL-Assisted Voice Creation
                - **RL Mode ON**  → the agent picks gender/pitch/speed for you based on what worked best before
                - **RL Mode OFF** → you control the sliders manually
                - After listening, **rate the output** → agent learns and improves
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        vc_text = gr.Textbox(
                            label="Input Text", lines=4,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters.",
                        )

                        use_rl = gr.Checkbox(
                            label="🤖 Let RL Agent choose voice parameters",
                            value=True,
                        )

                        with gr.Group():
                            gr.Markdown("*Sliders are used when RL Mode is OFF*")
                            vc_gender = gr.Radio(
                                choices=["male", "female"], value="male", label="Gender"
                            )
                            vc_pitch = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3, label="Pitch"
                            )
                            vc_speed = gr.Slider(
                                minimum=1, maximum=5, step=1, value=3, label="Speed"
                            )

                        create_btn = gr.Button("▶ Create Voice", variant="primary")

                    with gr.Column(scale=2):
                        vc_audio  = gr.Audio(label="Generated Audio (watermarked 🔐)", autoplay=True)
                        rl_note   = gr.Textbox(label="RL Agent Decision", interactive=False, lines=2)
                        rate_note = gr.Textbox(label="Rating", interactive=False, lines=1)

                        gr.Markdown("### ⭐ Rate this output")
                        gr.Markdown("*Your rating trains the RL agent to do better next time*")

                        with gr.Row():
                            btn1 = gr.Button("⭐ 1")
                            btn2 = gr.Button("⭐⭐ 2")
                            btn3 = gr.Button("⭐⭐⭐ 3")
                            btn4 = gr.Button("⭐⭐⭐⭐ 4")
                            btn5 = gr.Button("⭐⭐⭐⭐⭐ 5", variant="primary")

                        rating_status = gr.Textbox(
                            label="Rating Status", interactive=False, lines=2
                        )

                create_btn.click(
                    voice_creation_rl,
                    inputs=[vc_text, use_rl, vc_gender, vc_pitch, vc_speed],
                    outputs=[vc_audio, rl_note, rate_note],
                )

                # Star rating buttons
                btn1.click(fn=lambda: submit_rating(1), outputs=rating_status)
                btn2.click(fn=lambda: submit_rating(2), outputs=rating_status)
                btn3.click(fn=lambda: submit_rating(3), outputs=rating_status)
                btn4.click(fn=lambda: submit_rating(4), outputs=rating_status)
                btn5.click(fn=lambda: submit_rating(5), outputs=rating_status)

            # ── Tab 3: RL Stats ───────────────────────────────────────────────
            with gr.TabItem("📊 RL Agent Stats"):
                gr.Markdown("""
                ### What the RL Agent has learned
                Shows the best voice parameters discovered so far for each type of text.
                The more you rate, the smarter the agent gets.
                """)
                stats_box    = gr.Textbox(label="Agent Stats", lines=20, interactive=False)
                refresh_btn  = gr.Button("🔄 Refresh Stats")
                refresh_btn.click(fn=get_rl_stats, outputs=stats_box)

                gr.Markdown("""
                **How to read the table:**
                - **State** = text type (e.g. `short_positive` = short happy text)
                - **Best Action** = `gender_pitch_speed` the agent recommends
                - **Q-value** = learned score (+ve = good, -ve = bad)

                | State pattern | Meaning |
                |---|---|
                | `short_positive` | Short text with happy/excited tone |
                | `medium_negative` | Medium text with sad/angry tone |
                | `long_neutral` | Long neutral text (articles, info) |
                """)

            # ── Tab 4: Verify Watermark ───────────────────────────────────────
            with gr.TabItem("🔍 Verify Watermark"):
                gr.Markdown("""
                ### Check if audio was made by this system
                | | Who can check | Proves |
                |---|---|---|
                | **Public** | Anyone | Audio is AI-generated |
                | **Private** | Only you | Audio came from THIS model |
                """)
                v_file   = gr.Audio(type="filepath", label="Upload Audio File")
                v_btn    = gr.Button("🔍 Check Watermarks", variant="primary")
                v_public = gr.Textbox(label="Public Watermark",  lines=4, interactive=False)
                v_priv   = gr.Textbox(label="Private Watermark", lines=4, interactive=False)
                v_btn.click(verify_audio, inputs=v_file, outputs=[v_public, v_priv])

    return demo


def parse_arguments():
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument("--model_dir",    type=str, default="pretrained_models/Spark-TTS-0.5B")
    parser.add_argument("--device",       type=int, default=0)
    parser.add_argument("--server_name",  type=str, default="0.0.0.0")
    parser.add_argument("--server_port",  type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    demo = build_ui(model_dir=args.model_dir, device=args.device)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)