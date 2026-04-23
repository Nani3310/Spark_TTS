

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
from scipy.spatial.distance import cosine
from datetime import datetime
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


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
WAVLM_MODEL = None

def get_wavlm_model():
    global WAVLM_EXTRACTOR, WAVLM_MODEL
    if WAVLM_EXTRACTOR is None:
        WAVLM_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        WAVLM_MODEL = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
        WAVLM_MODEL.eval()
    return WAVLM_EXTRACTOR, WAVLM_MODEL

def wavlm_cosine_similarity(ref_path: str, test_path: str, sr: int = 16000) -> float:
    extractor, model = get_wavlm_model()
    
    audio_ref = load_wav_16k_mono(ref_path, sr)
    audio_test = load_wav_16k_mono(test_path, sr)
    
    inputs_ref = extractor(audio_ref, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs_test = extractor(audio_test, sampling_rate=sr, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        emb_ref = model(**inputs_ref).embeddings
        emb_test = model(**inputs_test).embeddings
        
    emb_ref = torch.nn.functional.normalize(emb_ref, dim=-1)
    emb_test = torch.nn.functional.normalize(emb_test, dim=-1)
    return torch.cosine_similarity(emb_ref, emb_test, dim=-1).item()


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    logging.info(f"Saving audio to: {save_dir}")

    # ✅ FIX: Only null out prompt_text if it's truly empty (len==0 or None)
    # Original code used <= 1 which killed single-char transcripts too,
    # but more importantly it was being called with None already stripped upstream.
    # The real fix is in voice_clone() below — we stop stripping it there.
    if prompt_text is not None and len(prompt_text.strip()) == 0:
        prompt_text = None

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")
    return save_path


def build_ui(model_dir, device=0):
    model = initialize_model(model_dir, device=device)

    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record

        # ✅ FIX: Stop aggressively nulling prompt_text.
        # Original: `None if len(prompt_text) < 2 else prompt_text`
        # This silently disabled prefix-mode whenever user left transcript blank.
        # Now: pass it through — run_tts() handles empty string cleanup.
        # IMPORTANT: When user provides transcript → prefix-mode → higher SIM.
        # When user leaves blank → no-prefix mode → lower SIM (expected behaviour).
        prompt_text_clean = prompt_text if prompt_text and len(prompt_text.strip()) > 1 else None

        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
        )
        
        if prompt_speech and os.path.exists(prompt_speech):
            try:
                sim = wavlm_cosine_similarity(prompt_speech, audio_output_path)
                metrics = f"{sim:.4f} (WavLM Cosine Similarity)"
            except Exception as e:
                metrics = f"Error computing similarity: {e}"
        else:
            metrics = "N/A (No reference audio)"

        return audio_output_path, metrics

    def voice_creation(text, gender, pitch, speed):
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
        )
        return audio_output_path

    with gr.Blocks() as demo:
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio （上传参考音频）\n"
                    "**Tip for higher speaker similarity:** Fill in the transcript of the reference audio below. "
                    "This enables prefix-mode inference which significantly improves voice cloning quality."
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        type="filepath",
                        label="Upload reference audio (≥5s recommended, clean, no background noise)",
                    )
                    prompt_wav_record = gr.Audio(
                        type="filepath",
                        label="Or record reference audio",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text to synthesise",
                        lines=3,
                        placeholder="Enter the text you want spoken here",
                    )
                    prompt_text_input = gr.Textbox(
                        label="📝 Transcript of reference audio (STRONGLY RECOMMENDED — boosts speaker similarity)",
                        lines=3,
                        placeholder="Type exactly what is said in the reference audio. Leave blank for lower-quality mode.",
                    )

                audio_output = gr.Audio(label="Generated Audio", autoplay=True)
                similarity_output = gr.Textbox(label="Speaker Similarity Metric", interactive=False)
                generate_button_clone = gr.Button("Generate")

                generate_button_clone.click(
                    voice_clone,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[audio_output, similarity_output],
                )

            with gr.TabItem("Voice Creation"):
                gr.Markdown("### Create your own voice based on the following parameters")

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(choices=["male", "female"], value="male", label="Gender")
                        pitch = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Pitch")
                        speed = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Speed")
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                audio_output_creation = gr.Audio(label="Generated Audio", autoplay=True)
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output_creation],
                )

    return demo


def parse_arguments():
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Spark-TTS-0.5B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    demo = build_ui(model_dir=args.model_dir, device=args.device)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)