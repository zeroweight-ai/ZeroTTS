"""Gradio demo for ZeroTTS.

    pip install "zerotts[webui]"
    python webui/app.py
    python webui/app.py --model ./local_model_dir

Voice selection is a picker over the precomputed voice packs shipped with the
weights. There is no "upload a reference clip" control, because there is nothing
behind it — the voice encoder is not part of this release. See docs/VOICES.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_stream  # noqa: E402
import engine  # noqa: E402

MODE_VOICE = "voice"
MODE_UNCOND = "uncond"
MODE_CHOICES = [
    ("Voice pack", MODE_VOICE),
    ("Unconditioned (no voice)", MODE_UNCOND),
]

_mounted_apps: set = set()


def _ensure_stream_route(app) -> None:
    """Register audio_stream's route on `app` once. Idempotent, and a no-op
    before a server exists."""
    if app is None or id(app) in _mounted_apps:
        return
    _mounted_apps.add(id(app))
    audio_stream.mount(app)


def refresh_voices():
    voices = engine.list_voices()
    return gr.update(choices=voices, value=voices[0] if voices else None)


def on_voice_change(name):
    """Preview + description for the selected voice.

    Returns a path or None — never a missing file, which is what makes
    gr.Audio render an error box instead of an empty player.
    """
    return engine.voice_preview_path(name), engine.voice_info(name)


def select_sample_text(evt: gr.SelectData, sample_names):
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not sample_names or not (0 <= idx < len(sample_names)):
        return gr.update()
    return engine.get_sample_texts().get(sample_names[idx], "")


def refresh_history():
    files = engine.list_generated()
    return gr.update(samples=[[os.path.basename(f)] for f in files]), files


def play_selected(evt: gr.SelectData, file_list):
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if file_list and 0 <= idx < len(file_list):
        return file_list[idx]
    return None


def clear_players():
    """Blank both players as their own event (not part of the generate run) so a
    new generation can't be heard on top of the previous one's buffered audio."""
    return audio_stream.player_html(None), None


def generate_ui(text, voice_name, mode, max_chunk_sec, cfg_scale, temperature,
                topk, topp, repetition_penalty, eoa_extra_frames, file_list):
    """Streams audio out through webui/audio_stream.py — a single continuous WAV
    response — NOT through gr.Audio(streaming=True).

    That component is not a continuous waveform: Gradio turns every yielded chunk
    into its own HLS segment with an independent AAC encode, so each one carries
    encoder priming at the front and zero padding at the back. Concatenated, every
    chunk boundary clicks; and our first chunks are 1-4 codec frames (0.08-0.32 s),
    shorter than AAC's own priming, which is why the first chunk appears to repeat.
    The saved .wav is always fine — it is the transport that is broken.

    Outputs: (live player HTML, completed-file player, status, history dataset,
    history state, segments box).
    """
    use_voice = mode == MODE_VOICE
    if use_voice and not voice_name:
        yield (gr.update(), gr.update(), "Select a voice first.",
               gr.update(), file_list, gr.update())
        return
    if len(text or "") > engine.MAX_TEXT_CHARS:
        yield (gr.update(), gr.update(),
               f"Text too long ({len(text)} chars, max {engine.MAX_TEXT_CHARS}).",
               gr.update(), file_list, gr.update())
        return

    segments = engine.get_text_segments(text, max_chunk_sec=max_chunk_sec)
    segments_text = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(segments))
    yield gr.update(), gr.update(), "Generating…", gr.update(), file_list, segments_text

    sample_rate = engine.get_sample_rate()
    sid = audio_stream.open_stream(sample_rate)
    # Show the player before the first chunk exists: the route blocks until audio
    # arrives, so the browser connects and starts buffering right away.
    yield (audio_stream.player_html(sid), gr.update(), "Generating…",
           gr.update(), file_list, gr.update())

    result: dict = {}
    n_samples = 0
    try:
        try:
            for _sr, chunk in engine.generate_stream(
                text=text, voice_name=voice_name, max_chunk_sec=max_chunk_sec,
                cfg_scale=float(cfg_scale), audio_temperature=temperature,
                audio_topk=int(topk), audio_topp=topp,
                audio_repetition_penalty=repetition_penalty,
                eoa_extra_frames=int(eoa_extra_frames), use_voice=use_voice,
                result=result,
            ):
                audio_stream.push(sid, chunk)
                n_samples += chunk.shape[0]
                yield (gr.update(), gr.update(),
                       f"Generating… {n_samples / sample_rate:.1f}s",
                       gr.update(), file_list, gr.update())
        except Exception as exc:
            yield gr.update(), gr.update(), f"Error: {exc}", gr.update(), file_list, gr.update()
            return
    finally:
        # Ends the HTTP response cleanly, including when the run is cancelled by
        # the Stop button (the generator is closed, which lands us here).
        audio_stream.close(sid)

    files = engine.list_generated()
    saved = result.get("path")
    yield (
        gr.update(),
        saved,
        f"Done — {n_samples / sample_rate:.1f}s. Saved to {saved}" if saved else "Done.",
        gr.update(samples=[[os.path.basename(f)] for f in files]),
        files,
        gr.update(),
    )


with gr.Blocks(title="ZeroTTS") as demo:
    gr.Markdown(
        "# ZeroTTS\n"
        "Vietnamese text-to-speech on ONNX Runtime — no PyTorch, no GPU."
    )

    history_state = gr.State([])
    sample_names_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                voice_dropdown = gr.Dropdown(choices=[], label="Voice", value=None)
                refresh_voices_btn = gr.Button("Refresh", scale=0)
            voice_meta = gr.Markdown("")
            voice_preview = gr.Audio(label="Voice preview", interactive=False)

            text_box = gr.Textbox(
                label=f"Text (max {engine.MAX_TEXT_CHARS} characters)",
                lines=10, max_lines=25,
                placeholder="Nhập văn bản tiếng Việt…",
            )
            sample_texts_dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[],
                label="Sample texts (webui/test_samples.txt — click to fill the box)",
            )
            gr.Markdown(
                "<sub>Punctuation is normalized before synthesis: `;` becomes a comma, "
                "and a line break becomes a sentence stop unless the line already ends "
                "in punctuation. The model is trained on transcribed speech, which has "
                "neither, and the tokenizer collapses whitespace — so an un-rewritten "
                "line break would simply vanish.</sub>"
            )

            mode_radio = gr.Radio(
                choices=MODE_CHOICES, value=MODE_VOICE, label="Speaker conditioning",
                info="Voice pack: prepends the selected voice's latents to the "
                     "sequence — the model's only speaker conditioning. "
                     "Unconditioned: the learned no-reference prefix, a voice the "
                     "model picks itself and does not keep consistent between "
                     "segments; a quality check only.",
            )
            cfg_slider = gr.Slider(
                1.0, 4.0, value=1.0, step=0.1, label="CFG scale (voice guidance)",
                info="1.0 = off: one forward pass per frame. Above 1.0 runs the "
                     "conditional and unconditional branches side by side and "
                     "extrapolates away from the unconditional one — stronger identity "
                     "at roughly 2x the cost. Ignored in unconditioned mode.",
            )

            with gr.Accordion("Advanced options", open=False):
                chunk_sec_slider = gr.Slider(5, 25, value=15, step=1,
                                             label="Max chunk length (seconds)")
                temperature_slider = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                               label="Temperature")
                topk_slider = gr.Slider(1, 200, value=25, step=1, label="Top-k")
                topp_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
                repetition_penalty_slider = gr.Slider(
                    1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty",
                    info="Penalizes audio codes already used in this segment. "
                         "1.2 is the benchmarked default; 1.0 disables it and "
                         "measurably raises WER.",
                )
                eoa_extra_slider = gr.Slider(
                    0, 4, value=1, step=1, label="Tail frames after stop",
                    info="Frames kept past the model's stop signal (0.08s each). The "
                         "stop token and that frame's audio are decoded together, so "
                         "the first is already computed — keeping it preserves the "
                         "last phone's release instead of cutting it mid-decay.",
                )

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop")

            gen_status = gr.Textbox(label="Status", interactive=False)
            segments_box = gr.Textbox(
                label="Segments sent to the model (post-chunking, post-cleanup)",
                interactive=False, lines=6,
            )
            # A plain <audio> fed by our own continuous-WAV route rather than
            # gr.Audio(streaming=True) — see generate_ui's docstring and
            # webui/audio_stream.py.
            gr.Markdown("**Output (plays as it generates)**")
            live_player = gr.HTML(value=audio_stream.player_html(None))
            completed_audio = gr.Audio(
                label="Output (complete file — seekable)", interactive=False,
                autoplay=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### History")
            refresh_history_btn = gr.Button("Refresh history")
            history_dataset = gr.Dataset(
                components=[gr.Textbox(visible=False)], samples=[],
                label="Click to play",
            )
            history_player = gr.Audio(label="Playback", interactive=False)

            gr.Markdown(
                "---\n"
                "**Voice cloning** is not available in the open-source release — "
                "the voice encoder is not published. To get latents for your own "
                "speaker, see [zeroweight.ai](https://zeroweight.ai)."
            )

    refresh_voices_btn.click(fn=refresh_voices, outputs=[voice_dropdown]).then(
        fn=on_voice_change, inputs=[voice_dropdown],
        outputs=[voice_preview, voice_meta],
    )
    voice_dropdown.change(fn=on_voice_change, inputs=[voice_dropdown],
                          outputs=[voice_preview, voice_meta])

    gen_event = generate_btn.click(
        fn=clear_players, outputs=[live_player, completed_audio],
    ).then(
        fn=generate_ui,
        inputs=[text_box, voice_dropdown, mode_radio, chunk_sec_slider, cfg_slider,
                temperature_slider, topk_slider, topp_slider,
                repetition_penalty_slider, eoa_extra_slider, history_state],
        outputs=[live_player, completed_audio, gen_status, history_dataset,
                 history_state, segments_box],
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

    refresh_history_btn.click(fn=refresh_history,
                              outputs=[history_dataset, history_state])
    history_dataset.select(fn=play_selected, inputs=[history_state],
                           outputs=[history_player])
    sample_texts_dataset.select(fn=select_sample_text, inputs=[sample_names_state],
                                outputs=[text_box])

    def _on_load():
        # Blocks is one shared graph across page loads, and this may have been
        # started by something other than __main__ (plain demo.launch(), `gradio
        # app.py`, an external mount) — the live-audio route has to exist on
        # whatever FastAPI app is actually serving us, or the player 404s.
        _ensure_stream_route(getattr(demo, "app", None))
        voices = engine.list_voices()
        first = voices[0] if voices else None
        files = engine.list_generated()
        sample_names = list(engine.get_sample_texts())
        preview, meta = on_voice_change(first)
        return (
            gr.update(choices=voices, value=first),
            preview, meta,
            gr.update(samples=[[os.path.basename(f)] for f in files]), files,
            gr.update(samples=[[n] for n in sample_names]), sample_names,
        )

    demo.load(
        fn=_on_load,
        outputs=[voice_dropdown, voice_preview, voice_meta,
                 history_dataset, history_state,
                 sample_texts_dataset, sample_names_state],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=engine.DEFAULT_MODEL)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    engine.set_model(args.model)

    # Mounted onto our own FastAPI app rather than demo.launch(), so the live
    # audio route is registered before the server starts. Gradio goes at "/" —
    # audio_stream.STREAM_ROUTE assumes that.
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    _ensure_stream_route(app)
    app = gr.mount_gradio_app(app, demo.queue(), path="/")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
