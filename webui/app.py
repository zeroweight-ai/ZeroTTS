"""Gradio demo for ZeroTTS.

    pip install "zerotts[webui]"
    python webui/app.py

Voice selection is a picker over the precomputed voice packs shipped with the
weights. There is no "upload a reference clip" control, because there is nothing
behind it: the voice encoder that would turn a clip into latents is not part of
this release. See the README.
"""

from __future__ import annotations

import argparse
import os
import time

import gradio as gr
import numpy as np

from zerotts import ZeroTTS
from zerotts.chunking import chunk_text, clean_segment_punctuation, normalize_punctuation

MAX_TEXT_CHARS = 5000

SAMPLE_TEXTS = {
    "Chào hỏi": "Xin chào các bạn, mình là ZeroTTS. Rất vui được gặp mọi người.",
    "Tin tức": "Theo báo cáo mới nhất, lạm phát năm nay ở mức 3,2%, thấp hơn dự báo "
               "của các chuyên gia kinh tế.",
    "Kể chuyện": "Ngày xửa ngày xưa, ở một ngôi làng nhỏ ven sông, có một cô bé rất "
                 "thích ngắm những cánh diều bay trên bầu trời chiều.",
    "Code-switch": "Mình đang dùng một model text to speech chạy real-time trên CPU, "
                   "không cần GPU luôn.",
}

_tts: ZeroTTS | None = None


def get_tts() -> ZeroTTS:
    if _tts is None:
        raise RuntimeError("model not loaded")
    return _tts


def _segments(text: str, chunk: bool, max_chunk_sec: float) -> list:
    if not chunk:
        return [text.strip()]
    segs = [clean_segment_punctuation(s)
            for s in chunk_text(normalize_punctuation(text), max_chunk_sec=max_chunk_sec)]
    return [s for s in segs if s]


def generate(text, voice, use_voice, cfg_scale, temperature, topk, topp,
             repetition_penalty, chunk, max_chunk_sec, seed, progress=gr.Progress()):
    """Streaming generation — yields growing audio so playback can start early."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text first.")
    if len(text) > MAX_TEXT_CHARS:
        raise gr.Error(f"Text is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.")

    tts = get_tts()
    if seed is not None and int(seed) >= 0:
        np.random.seed(int(seed))

    voice_arg = voice if (use_voice and voice) else None
    segments = _segments(text, chunk, max_chunk_sec)
    sr = tts.sample_rate
    gap = np.zeros(int(0.15 * sr), dtype=np.float32)

    collected: list = []
    t0 = time.perf_counter()
    for i, seg in enumerate(segments):
        progress((i / max(1, len(segments))), desc=f"Segment {i + 1}/{len(segments)}")
        if i:
            collected.append(gap)
        for chunk_audio in tts.synthesize_stream(
            seg, voice=voice_arg, cfg_scale=cfg_scale,
            audio_temperature=temperature, audio_topk=int(topk), audio_topp=topp,
            audio_repetition_penalty=repetition_penalty,
        ):
            collected.append(chunk_audio.reshape(-1))
            yield (sr, np.concatenate(collected)), gr.update()

    audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    elapsed = time.perf_counter() - t0
    dur = len(audio) / sr
    status = (f"{dur:.2f}s audio in {elapsed:.2f}s — {dur / elapsed:.1f}x realtime, "
              f"{len(segments)} segment(s)")
    yield (sr, audio), status


def build_ui(tts: ZeroTTS) -> gr.Blocks:
    voices = tts.list_voices()
    index = {}
    for name in voices:
        v = tts.load_voice(name)
        index[name] = v

    with gr.Blocks(title="ZeroTTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ZeroTTS\n"
            "Vietnamese text-to-speech running on ONNX Runtime — no PyTorch, CPU is fine.\n"
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text", lines=6, max_lines=20,
                    placeholder="Nhập văn bản tiếng Việt…",
                    value=SAMPLE_TEXTS["Chào hỏi"])
                with gr.Row():
                    sample = gr.Dropdown(
                        label="Sample text", choices=list(SAMPLE_TEXTS), value=None,
                        scale=2)
                    generate_btn = gr.Button("Generate", variant="primary", scale=1)

            with gr.Column(scale=2):
                use_voice = gr.Checkbox(
                    label="Use a voice", value=bool(voices),
                    info="Off = the model's unconditional voice (not stable across runs)")
                voice = gr.Dropdown(
                    label="Voice", choices=voices,
                    value=voices[0] if voices else None,
                    interactive=bool(voices))
                preview = gr.Audio(label="Voice preview", interactive=False,
                                   visible=bool(voices))
                if not voices:
                    gr.Markdown(
                        "> No voice packs found in this model directory.")
                gr.Markdown(
                    "Voice cloning is not available in the open-source release — the "
                    "voice encoder is not published. To get latents for your own "
                    "speaker, see [zeroweight.ai](https://zeroweight.ai)."
                )

        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                cfg_scale = gr.Slider(1.0, 3.0, value=1.0, step=0.1, label="CFG scale",
                                      info=">1 pushes toward the voice, 2x slower")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                seed = gr.Number(value=-1, label="Seed", precision=0,
                                 info="-1 for random")
            with gr.Row():
                topk = gr.Slider(1, 200, value=25, step=1, label="Top-k")
                topp = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty",
                    info="1.2 is the benchmarked default; 1.0 raises WER")
            with gr.Row():
                chunk = gr.Checkbox(label="Split long text into segments", value=True)
                max_chunk_sec = gr.Slider(5, 30, value=15, step=1,
                                          label="Max segment length (s)")

        audio_out = gr.Audio(label="Output", streaming=False, autoplay=True)
        status = gr.Markdown("")

        def _preview(name):
            v = index.get(name)
            return gr.update(value=v.preview_path if v and v.preview_path else None)

        voice.change(_preview, inputs=voice, outputs=preview)
        sample.change(lambda k: SAMPLE_TEXTS.get(k, ""), inputs=sample, outputs=text)
        generate_btn.click(
            generate,
            inputs=[text, voice, use_voice, cfg_scale, temperature, topk, topp,
                    repetition_penalty, chunk, max_chunk_sec, seed],
            outputs=[audio_out, status],
        )

        if voices:
            demo.load(_preview, inputs=voice, outputs=preview)

    return demo


def main() -> None:
    global _tts
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ZEROTTS_MODEL", "zeroweight-ai/ZeroTTS"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model} …")
    _tts = ZeroTTS.from_pretrained(args.model, intra_op_num_threads=args.threads)
    print(f"Ready. Voices: {_tts.list_voices() or '(none)'}")
    build_ui(_tts).queue().launch(server_name=args.host, server_port=args.port,
                                  share=args.share)


if __name__ == "__main__":
    main()
