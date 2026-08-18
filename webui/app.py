"""Gradio demo for ZeroTTS.

    pip install "zerotts[webui]"
    python webui/app.py
    python webui/app.py --model ./local_model_dir

The page is deliberately split in two. The BASIC view is text → voice → player
and nothing else, so someone who just wants to hear a sentence never meets a
sampler knob. Everything technical — sampling parameters, the run log, the
segments actually sent to the model — lives under "Tuỳ chọn nâng cao".

Voice selection is a picker over the precomputed voice packs shipped with the
weights. There is no "upload a reference clip" control, because there is nothing
behind it — the voice encoder is not part of this release. See docs/VOICES.md.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_stream  # noqa: E402
import engine  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
BANNER_PATH = os.path.join(_ROOT, "docs", "assets", "banner.png")

DEFAULT_TEXT = "Xin chào tất cả mọi người. Giọng nói này được tạo ra bởi ZeroTTS."

MODE_VOICE = "voice"
MODE_UNCOND = "uncond"
MODE_CHOICES = [
    ("Voice pack", MODE_VOICE),
    ("Unconditioned (no voice)", MODE_UNCOND),
]

_mounted_apps: set = set()


# ── chrome ───────────────────────────────────────────────────────────────────

def banner_html() -> str:
    """The banner, inlined as a data URI so it needs no static route and no
    `allowed_paths` entry. Degrades to a wordmark if the file is missing."""
    try:
        with open(BANNER_PATH, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return "<div class='zt-banner zt-banner-text'><span>Zero</span>TTS</div>"
    return f"<div class='zt-banner'><img src='{src}' alt='ZeroTTS' /></div>"


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
)

# One brand ramp (the banner's orange) plus a violet accent, then a handful of
# structural overrides. The two list-shaped panels (history, templates) are
# gr.Dataset with two hidden Textbox columns, which Gradio renders as a <table>
# of <tr class="tr-body"> — the .zt-list rules below turn each of those rows
# into a two-line card: title cell on top, preview cell under it. Two columns
# and not one because the Dataset frontend cuts every CELL at 60 characters
# (Example-*.js `slice(0, 60) + "..."`), so a title and a readable preview do
# not fit in a single one.
CSS = """
:root, .gradio-container {
  --zt-brand: #f4530c;      /* the banner's orange */
  --zt-brand-2: #ff8a3d;
  --zt-soft: rgba(244, 83, 12, .09);
}
/* The wash goes full-bleed, not on the (centred, max-width) container where it
   would stop dead at the edges and read as a stray rectangle. <gradio-app> is
   the element that paints the page colour — body sits behind it. */
gradio-app {
  background-image:
    radial-gradient(1200px 560px at 4% -8%, rgba(244,83,12,.20), transparent 62%),
    radial-gradient(1000px 500px at 99% 0%, rgba(124,58,237,.18), transparent 64%) !important;
  background-attachment: fixed !important;
  background-repeat: no-repeat !important;
}
/* <gradio-app> lays its child out with flex and no justify-content, so the
   capped-width container sticks to the left edge until it is given auto
   margins — max-width alone does not centre it. */
.gradio-container {
  max-width: 1180px !important; margin: 0 auto !important;
  background: transparent !important;
}
footer { display: none !important; }

/* banner */
.zt-banner { margin: 0 auto .25rem; text-align: center; }
.zt-banner img {
  width: 100%; max-width: 560px; height: auto; display: block; margin: 0 auto;
  border-radius: 18px;
}
.zt-banner-text { font-size: 2.4rem; font-weight: 800; letter-spacing: -.02em; }
.zt-banner-text span { color: var(--zt-brand); }
.zt-headline { text-align: center; margin: .9rem auto 1.1rem !important; max-width: 62ch; }
.zt-headline h1 {
  font-size: 1.3rem !important; font-weight: 700 !important;
  line-height: 1.35 !important; letter-spacing: -.01em;
  margin: 0 !important; padding: 0 !important;
}

/* cards */
.zt-card {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 20px !important;
  padding: 1.15rem 1.15rem 1.25rem !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 10px 30px -22px rgba(20, 20, 40, .55);
}
/* Section titles read as headings, not as body copy: heavier, slightly larger,
   and led by a short brand bar so the eye finds the start of each card. */
.zt-card-title {
  display: flex; align-items: center; gap: .55rem;
  margin: 0 0 .35rem !important;
}
.zt-card-title p, .zt-card-title h1, .zt-card-title h2, .zt-card-title h3 {
  font-weight: 800 !important; font-size: 1.14rem !important;
  letter-spacing: -.015em; line-height: 1.3 !important;
  margin: 0 !important; padding: 0 !important;
}
.zt-card-title::before {
  content: ""; flex: none; width: .28rem; height: 1.15rem; border-radius: 999px;
  background: linear-gradient(180deg, var(--zt-brand-2), var(--zt-brand));
}
.zt-hint {
  color: var(--body-text-color-subdued); font-size: .84rem;
  margin: 0 0 .55rem !important;
}

/* the primary action */
#zt-generate {
  background: linear-gradient(135deg, var(--zt-brand-2), var(--zt-brand)) !important;
  border: none !important; color: #fff !important;
  font-weight: 700 !important; font-size: 1.02rem !important;
  padding: .8rem 1rem !important;
  box-shadow: 0 12px 24px -14px rgba(244, 83, 12, .95);
}
#zt-generate:hover { filter: brightness(1.06); }
.zt-actions { margin-top: .35rem; align-items: stretch; }
.zt-actions button { min-height: 3rem; }

/* the run-detail accordion nested inside the text card: a quiet disclosure,
   not a second panel competing with the card it sits in */
.zt-inline-accordion {
  border: none !important; background: transparent !important;
  margin-top: .5rem !important; padding: 0 !important;
}
.zt-inline-accordion > .label-wrap {
  padding: 0 !important; font-size: .85rem;
  color: var(--body-text-color-subdued) !important;
}
/* the block's own status wrap keeps its `hide` class even when the accordion is
   open; with the accordion's padding gone it shows through as a stray scrollbar */
.zt-inline-accordion > .wrap.hide { display: none !important; }

/* list-shaped datasets (history, templates) */
.zt-list .table-wrap { border: none !important; overflow: visible !important; }
.zt-list table { width: 100% !important; border-collapse: separate; }
.zt-list thead, .zt-list .tr-head { display: none !important; }
.zt-list tbody { display: flex; flex-direction: column; gap: .5rem; }
.zt-list .tr-body {
  display: flex !important; flex-direction: column; align-items: stretch;
  flex: none;  /* rows are flex items now; without this they squash */
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 14px !important; background: var(--background-fill-secondary) !important;
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
.zt-list .tr-body:hover {
  border-color: var(--zt-brand) !important; background: var(--zt-soft) !important;
  transform: translateY(-1px);
}
.zt-list td {
  border: none !important; text-align: left !important;
  padding: .1rem .85rem !important; max-width: none !important;
  white-space: pre-wrap;  /* the separators in a row title are real spaces */
}
.zt-list td:first-child {
  padding-top: .6rem !important;
  font-weight: 700; font-size: .93rem; color: var(--body-text-color);
}
.zt-list td:last-child {
  padding-bottom: .6rem !important;
  font-size: .84rem; color: var(--body-text-color-subdued);
}
.zt-list td > * { text-align: left !important; }
.zt-list .paginate { justify-content: flex-start; font-size: .8rem; }
#zt-history tbody { max-height: 27rem; overflow-y: auto; overflow-x: hidden; }
#zt-templates tbody { flex-direction: row; flex-wrap: wrap; }
#zt-templates .tr-body { width: calc(50% - .25rem); }
@media (max-width: 820px) { #zt-templates .tr-body { width: 100%; } }

/* players */
.zt-player audio { width: 100%; }
.zt-live { margin-top: .2rem; }
/* The live element is a plain <audio> with the browser's own controls (see
   audio_stream.py) — color-scheme is the only way to stop Chrome painting it
   bright white in the middle of the dark theme. */
.zt-live audio { width: 100%; }
.dark .zt-live audio { color-scheme: dark; }
.zt-foot {
  text-align: center; font-size: .84rem; line-height: 1.5;
  color: var(--body-text-color-subdued);
  margin: .9rem 0 0 !important;
}
"""


# Gradio 6 moved `theme`/`css` off the Blocks constructor and onto
# launch()/mount_gradio_app(); 4.x and 5.x only accept them on the constructor.
# pyproject supports gradio>=4.44, so decide per install rather than pinning.
_STYLE = {"theme": THEME, "css": CSS}
_STYLE_ON_MOUNT = "css" in inspect.signature(gr.mount_gradio_app).parameters
_STYLE_ON_BLOCKS = {} if _STYLE_ON_MOUNT else _STYLE


def _ensure_stream_route(app) -> None:
    """Register audio_stream's route on `app` once. Idempotent, and a no-op
    before a server exists."""
    if app is None or id(app) in _mounted_apps:
        return
    _mounted_apps.add(id(app))
    audio_stream.mount(app)


# ── voices ───────────────────────────────────────────────────────────────────

def refresh_voices():
    """Dropdown choices. Prefers `maichi` as the default so a fresh page load
    always starts on the same voice the README and samples use."""
    choices = engine.voice_choices(None)
    names = [v for _, v in choices]
    default = "maichi" if "maichi" in names else (names[0] if names else None)
    return gr.update(choices=choices, value=default)


def on_voice_change(name):
    """Preview + description for the selected voice.

    Returns a path or None — never a missing file, which is what makes
    gr.Audio render an error box instead of an empty player.
    """
    return engine.voice_preview_path(name), engine.voice_info(name)


# ── templates ────────────────────────────────────────────────────────────────

# Gradio's Dataset frontend cuts every cell at 60 characters and appends "...",
# so shorten here instead and use a nicer ellipsis.
CELL_CHARS = 57


def _shorten(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 1].rstrip() + "…"


def _template_rows(samples: dict) -> list:
    """One card per template: its name, then a preview of the text itself —
    which is what someone picking a template actually wants to see."""
    return [[name, _shorten(text)] for name, text in samples.items()]


def select_sample_text(evt: gr.SelectData, sample_names):
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not sample_names or not (0 <= idx < len(sample_names)):
        return gr.update()
    return engine.get_sample_texts().get(sample_names[idx], "")


# ── history ──────────────────────────────────────────────────────────────────

def _history_rows(paths: list) -> list:
    """One card per saved file: voice + when + length, then the text that was
    spoken. Clicking a card loads it into the main player."""
    rows = []
    for path in paths:
        meta = engine.generated_meta(path)
        head = f"🔊  {engine.voice_display_name(meta['voice']) or meta['voice']}"
        head += f"  ·  {meta['when']}"
        if meta["seconds"]:
            head += f"  ·  {meta['seconds']:.0f}s"
        rows.append([_shorten(head), _shorten(meta["text"]) or "—"])
    return rows


def refresh_history():
    files = engine.list_generated()
    return gr.update(samples=_history_rows(files)), files


def play_selected(evt: gr.SelectData, file_list):
    """History click → the main output player, so there is exactly one place
    audio comes from on this page."""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if file_list and 0 <= idx < len(file_list):
        return file_list[idx]
    return None


# ── generation ───────────────────────────────────────────────────────────────

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
        yield (gr.update(), gr.update(), "Hãy chọn một giọng đọc trước.",
               gr.update(), file_list, gr.update())
        return
    if len(text or "") > engine.MAX_TEXT_CHARS:
        yield (gr.update(), gr.update(),
               f"Văn bản quá dài ({len(text)} ký tự, tối đa {engine.MAX_TEXT_CHARS}).",
               gr.update(), file_list, gr.update())
        return

    segments = engine.get_text_segments(text, max_chunk_sec=max_chunk_sec)
    segments_text = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(segments))
    yield gr.update(), gr.update(), "Đang tạo…", gr.update(), file_list, segments_text

    sample_rate = engine.get_sample_rate()
    sid = audio_stream.open_stream(sample_rate)
    # Show the player before the first chunk exists: the route blocks until audio
    # arrives, so the browser connects and starts buffering right away.
    yield (audio_stream.player_html(sid), gr.update(), "Đang tạo…",
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
                       f"Đang tạo… {n_samples / sample_rate:.1f}s",
                       gr.update(), file_list, gr.update())
        except Exception as exc:
            yield gr.update(), gr.update(), f"Lỗi: {exc}", gr.update(), file_list, gr.update()
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
        f"Xong — {n_samples / sample_rate:.1f}s. Đã lưu vào {saved}" if saved else "Xong.",
        gr.update(samples=_history_rows(files)),
        files,
        gr.update(),
    )


with gr.Blocks(title="ZeroTTS", **_STYLE_ON_BLOCKS) as demo:
    gr.HTML(banner_html())
    gr.Markdown(
        "# ZeroTTS - Chuyển văn bản tiếng Việt thành giọng nói tự nhiên, "
        "nhanh và realtime trên CPU",
        elem_classes="zt-headline",
    )

    history_state = gr.State([])
    sample_names_state = gr.State([])

    with gr.Row():
        # ── basic view: text → player, and nothing else ──────────────────────
        with gr.Column(scale=3):
            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Nhập văn bản", elem_classes="zt-card-title")
                gr.Markdown(
                    f"Tối đa {engine.MAX_TEXT_CHARS} ký tự. Chưa biết viết gì? "
                    "Chọn một mẫu câu ở bên dưới.",
                    elem_classes="zt-hint",
                )
                text_box = gr.Textbox(
                    value=DEFAULT_TEXT, label=None, show_label=False,
                    lines=7, max_lines=20, container=False,
                    placeholder="Nhập văn bản tiếng Việt…",
                )
                with gr.Row(elem_classes="zt-actions"):
                    generate_btn = gr.Button("🎙️  Tạo giọng nói", variant="primary",
                                             elem_id="zt-generate", scale=3)
                    stop_btn = gr.Button("Dừng", scale=1)

                # What the run did, next to the run itself rather than down in
                # the global settings accordion — it is about this text, not
                # about how the model is configured.
                with gr.Accordion("Xem chi tiết lần chạy", open=False,
                                  elem_classes="zt-inline-accordion"):
                    gen_status = gr.Textbox(label="Trạng thái", interactive=False)
                    segments_box = gr.Textbox(
                        label="Các đoạn được gửi tới mô hình "
                              "(sau khi tách câu và làm sạch)",
                        interactive=False, lines=6,
                    )

            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Nghe kết quả", elem_classes="zt-card-title")
                gr.Markdown("Phát ngay trong lúc đang tạo:", elem_classes="zt-hint")
                # A plain <audio> fed by our own continuous-WAV route rather than
                # gr.Audio(streaming=True) — see generate_ui's docstring and
                # webui/audio_stream.py.
                live_player = gr.HTML(value=audio_stream.player_html(None),
                                      elem_classes="zt-live")
                completed_audio = gr.Audio(
                    label="Bản hoàn chỉnh — tua và tải về được",
                    interactive=False, autoplay=False, elem_classes="zt-player",
                )
                gr.Markdown(
                    "**Nhân bản giọng nói (voice cloning)** không có trong bản mã "
                    "nguồn mở — bộ mã hoá giọng chưa được phát hành. Cần latents "
                    "cho giọng của riêng bạn? Xem "
                    "[zeroweight.ai](https://zeroweight.ai).",
                    elem_classes="zt-foot",
                )

        # ── voice picker, then the takes it produced ─────────────────────────
        with gr.Column(scale=2):
            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Chọn giọng đọc", elem_classes="zt-card-title")
                with gr.Row():
                    voice_dropdown = gr.Dropdown(choices=[], label=None,
                                                 show_label=False, value=None,
                                                 container=False, scale=5)
                    refresh_voices_btn = gr.Button("↻", scale=0, min_width=48)
                voice_meta = gr.Markdown("", elem_classes="zt-hint")
                voice_preview = gr.Audio(label="Nghe thử giọng", interactive=False,
                                         elem_classes="zt-player")

            with gr.Column(elem_classes="zt-card"):
                with gr.Row():
                    gr.Markdown("Đã tạo gần đây", elem_classes="zt-card-title")
                    refresh_history_btn = gr.Button("↻", scale=0, min_width=48)
                gr.Markdown("Bấm vào một mục để nghe lại ở trình phát chính.",
                            elem_classes="zt-hint")
                history_dataset = gr.Dataset(
                    components=[gr.Textbox(visible=False),
                                gr.Textbox(visible=False)], samples=[],
                    label=None, show_label=False, samples_per_page=20,
                    elem_id="zt-history", elem_classes="zt-list",
                )

    # ── templates, below the fold: name + a real preview of the text ─────────
    with gr.Column(elem_classes="zt-card"):
        gr.Markdown("Mẫu câu", elem_classes="zt-card-title")
        gr.Markdown("Bấm một mẫu để điền vào ô văn bản ở trên.",
                    elem_classes="zt-hint")
        sample_texts_dataset = gr.Dataset(
            components=[gr.Textbox(visible=False),
                        gr.Textbox(visible=False)], samples=[],
            label=None, show_label=False, samples_per_page=12,
            elem_id="zt-templates", elem_classes="zt-list",
        )

    # ── advanced view: everything technical, closed by default ──────────────
    with gr.Accordion("Tuỳ chọn nâng cao", open=False):
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
        with gr.Row():
            chunk_sec_slider = gr.Slider(5, 25, value=15, step=1,
                                         label="Max chunk length (seconds)")
            temperature_slider = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                           label="Temperature")
        with gr.Row():
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
        gr.Markdown(
            "<sub>Punctuation is normalized before synthesis: `;` becomes a comma, "
            "and a line break becomes a sentence stop unless the line already ends "
            "in punctuation. The model is trained on transcribed speech, which has "
            "neither, and the tokenizer collapses whitespace — so an un-rewritten "
            "line break would simply vanish.</sub>"
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
                           outputs=[completed_audio])
    sample_texts_dataset.select(fn=select_sample_text, inputs=[sample_names_state],
                                outputs=[text_box])

    def _on_load():
        # Blocks is one shared graph across page loads, and this may have been
        # started by something other than __main__ (plain demo.launch(), `gradio
        # app.py`, an external mount) — the live-audio route has to exist on
        # whatever FastAPI app is actually serving us, or the player 404s.
        _ensure_stream_route(getattr(demo, "app", None))
        voice_update = refresh_voices()
        default = voice_update["value"]
        files = engine.list_generated()
        samples = engine.get_sample_texts()
        preview, meta = on_voice_change(default)
        return (
            voice_update,
            preview, meta,
            gr.update(samples=_history_rows(files)), files,
            gr.update(samples=_template_rows(samples)), list(samples),
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
    app = gr.mount_gradio_app(app, demo.queue(), path="/",
                              **(_STYLE if _STYLE_ON_MOUNT else {}))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
