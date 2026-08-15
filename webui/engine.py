"""Backend for the Gradio demo: voice packs, long-form chunking, and streamed
generation. Kept UI-framework-agnostic so app.py just calls into here.

Speaker conditioning is a precomputed voice pack's latents, or the model's
learned unconditional prefix. There is no cloning — see zerotts.voices.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

from zerotts import ZeroTTS
from zerotts.chunking import (
    chunk_text,
    clean_segment_punctuation,
    load_text_samples,
    normalize_punctuation,
)
from zerotts.text_norm import normalize_vi_text

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_MODEL = os.environ.get("ZEROTTS_MODEL", "zeroweight-ai/ZeroTTS")
GENERATED_DIR = os.path.join(_ROOT, "outputs", "generated")
SAMPLE_TEXTS_PATH = os.path.join(_HERE, "test_samples.txt")
MAX_TEXT_CHARS = 5000

# Frames of canonical silence appended after every text segment during streaming
# generation, so the codec's causal decoder (and the listener) gets a clean
# breath between segments instead of one butting against the next.
SILENCE_FRAMES_PER_CHUNK = 4

os.makedirs(GENERATED_DIR, exist_ok=True)

_tts: ZeroTTS | None = None
_silence_frame: np.ndarray | None = None
_model_id = DEFAULT_MODEL


def set_model(model_id: str) -> None:
    global _model_id
    _model_id = model_id


def get_tts() -> ZeroTTS:
    global _tts
    if _tts is None:
        print(f"Loading ZeroTTS from {_model_id} ...")
        _tts = ZeroTTS.from_pretrained(_model_id)
    return _tts


def get_sample_rate() -> int:
    """Output sample rate. Loads the model on first call (which generation would
    do a moment later anyway) — the streaming player needs it to write its WAV
    header before the first chunk exists."""
    return int(get_tts().sample_rate)


def _get_silence_frame(tts: ZeroTTS) -> np.ndarray:
    """The codec's canonical silence RVQ frame, (1, num_codebooks) int64.

    Shipped as silence_frame.npy rather than derived here: deriving it means
    encoding digital silence and taking the per-codebook mode, which needs the
    codec ENCODER — deliberately not part of this release. It is precomputed at
    build time from the same codec weights.

    Older model directories may predate the file; falling back to zeros is wrong
    (code 0 is a real code, not silence) and would inject a burst of noise
    between segments, so we skip the padding entirely instead.
    """
    global _silence_frame
    if _silence_frame is None:
        path = tts.model_dir / "silence_frame.npy"
        if not path.exists():
            return None  # caller skips inter-segment padding
        _silence_frame = np.load(path).astype(np.int64).reshape(1, -1)
    return _silence_frame


# ── voices ───────────────────────────────────────────────────────────────────

def list_voices() -> list:
    try:
        return get_tts().list_voices()
    except Exception:
        return []


def voice_preview_path(name: str) -> str | None:
    """Path to a voice's preview wav, or None. Never raises — the UI calls this
    on every dropdown change, and a missing preview is not an error."""
    if not name:
        return None
    try:
        voice = get_tts().load_voice(name)
    except Exception:
        return None
    path = voice.preview_path
    return path if path and os.path.isfile(path) else None


def voice_info(name: str) -> str:
    if not name:
        return ""
    try:
        v = get_tts().load_voice(name)
    except Exception as exc:
        return f"Could not load voice: {exc}"
    bits = [f"`{v.name}`", v.language]
    if v.description:
        bits.append(v.description)
    return " · ".join(bits)


# ── history ──────────────────────────────────────────────────────────────────

def list_generated() -> list:
    """Newest-first list of saved generated-audio paths."""
    if not os.path.isdir(GENERATED_DIR):
        return []
    files = [os.path.join(GENERATED_DIR, f)
             for f in os.listdir(GENERATED_DIR) if f.endswith(".wav")]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


# ── text ─────────────────────────────────────────────────────────────────────

def get_sample_texts() -> dict:
    """{name: text} from webui/test_samples.txt — hand-edited, so re-read every
    call rather than cached."""
    return load_text_samples(SAMPLE_TEXTS_PATH)


def get_text_segments(text: str, max_chunk_sec: float = 15.0,
                      normalize_numbers: bool = True) -> list:
    """Exactly the segments generate_stream will synthesize, after Vietnamese
    text normalization, chunking and per-segment punctuation cleanup — exposed
    so the UI can show what will actually be spoken before generation starts.

    ``normalize_numbers`` runs zerotts.text_norm.normalize_vi_text first,
    turning dates, clock times, versions, fractions, acronyms and numbers into
    spoken Vietnamese. It runs BEFORE chunking because an expansion is several
    times longer than what it replaces, and the chunk budget has to size the
    text the model actually receives.

    Turn it off for non-Vietnamese text: the expansions are Vietnamese words, so
    "3/4" in an English sentence would come out "ba trên bốn".
    """
    if normalize_numbers:
        text = normalize_vi_text(text)
    raw = chunk_text(normalize_punctuation(text), max_chunk_sec=max_chunk_sec)
    return [c for c in (clean_segment_punctuation(x) for x in raw) if c]


# ── generation ───────────────────────────────────────────────────────────────

def generate_stream(
    text: str,
    voice_name: str | None,
    max_chunk_sec: float = 15.0,
    cfg_scale: float = 1.0,
    audio_temperature: float = 0.8,
    audio_topk: int = 25,
    audio_topp: float = 0.95,
    audio_repetition_penalty: float = 1.2,
    min_frames: int = 4,
    max_frames: int = 1500,
    max_chunk_frames: int = 16,
    eoa_extra_frames: int = 1,
    use_voice: bool = True,
    normalize_numbers: bool = True,
    result: dict | None = None,
):
    """Generator yielding (sample_rate, int16 ndarray) chunks for streaming.

    The text is punctuation-normalized and sentence-chunked into segments, and
    each segment is its own generate call — the architecture re-primes the
    global transformer's [voice | soa] prefix fresh per call.

    But the CODEC is a single continuous streaming session shared across ALL
    segments: one streaming decoder is opened up front and every segment's frames
    go through it, so the causal audio decoder's KV cache never resets and there
    is no cold-start discontinuity at a segment boundary. Getting this wrong —
    one decoder per segment — is audible as a click between segments.

    The chunk-size ramp (1, 2, 4 … frames per decode call) only buys
    time-to-first-audio, which only matters once, so it applies to the first
    segment only; later segments decode straight at ``max_chunk_frames``.

    On completion (including when the caller stops iterating early) the full
    audio is saved under GENERATED_DIR; pass a ``result`` dict to read back
    "path" and "n_samples" afterwards, since a generator cannot return a value
    through a plain for-loop.
    """
    if not text or not text.strip():
        raise ValueError("Please enter some text to synthesize.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text is too long ({len(text)} chars) — max {MAX_TEXT_CHARS}.")

    tts = get_tts()
    voice_emb = None
    if use_voice:
        if not voice_name:
            raise ValueError("Please select a voice first.")
        voice_emb = tts.resolve_voice(voice_name)
    else:
        # Guiding away from the unconditional branch when the conditional branch
        # IS the unconditional branch is a no-op that costs twice as much.
        cfg_scale = 1.0

    segments = get_text_segments(text, max_chunk_sec=max_chunk_sec,
                                 normalize_numbers=normalize_numbers)
    silence_frame = _get_silence_frame(tts)

    stream = tts.codec.streaming_decoder()
    all_chunks: list = []

    def _emit(frames: list):
        codes = np.stack(frames, axis=-1)          # (1, num_codebooks, n)
        arr = np.asarray(stream.decode_chunk(codes)).squeeze(0)
        all_chunks.append(arr)
        return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)

    try:
        for seg_idx, segment in enumerate(segments):
            target = 1 if seg_idx == 0 else max_chunk_frames  # ramp on the first only
            buf: list = []
            for frame_codes in tts._generate_frames(
                segment, min_frames, max_frames,
                voice_emb=voice_emb, cfg_scale=cfg_scale,
                audio_temperature=audio_temperature, audio_topk=audio_topk,
                audio_topp=audio_topp,
                audio_repetition_penalty=audio_repetition_penalty,
                eoa_extra_frames=eoa_extra_frames,
            ):
                buf.append(frame_codes)
                if len(buf) >= target:
                    chunk = _emit(buf)
                    buf = []
                    if seg_idx == 0:
                        target = min(max_chunk_frames, target * 2)
                    yield tts.sample_rate, chunk
            if buf:
                yield tts.sample_rate, _emit(buf)
            if silence_frame is not None:
                yield tts.sample_rate, _emit([silence_frame] * SILENCE_FRAMES_PER_CHUNK)
    finally:
        stream.close()
        full = np.concatenate(all_chunks) if all_chunks else np.zeros(0, dtype=np.float32)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = voice_name if use_voice else "uncond"
        out_path = os.path.join(GENERATED_DIR, f"{ts}_{tag or 'uncond'}.wav")
        sf.write(out_path, full, tts.sample_rate, subtype="PCM_16")
        with open(out_path + ".json", "w") as f:
            json.dump({"voice": voice_name if use_voice else None,
                       "cfg_scale": cfg_scale,
                       "text": text.strip().replace("\n", " ")[:200],
                       "created_at": ts}, f, indent=2, ensure_ascii=False)
        if result is not None:
            result["path"] = out_path
            result["n_samples"] = int(full.shape[0])
