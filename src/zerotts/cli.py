"""Command-line interface: ``zerotts say`` / ``zerotts voices`` / ``zerotts bench``."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import hub
from .audio import concat_with_silence
from .chunking import chunk_text, clean_segment_punctuation, normalize_punctuation
from .synthesizer import ZeroTTS
from .text_norm import normalize_vi_text


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=hub.DEFAULT_REPO_ID,
                   help="HF repo id or local model directory.")
    p.add_argument("--revision", default=None, help="HF revision to pin.")
    p.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads.")


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help=">1 guides toward the voice, at 2x the per-frame cost.")
    p.add_argument("--audio_temperature", type=float, default=0.8)
    p.add_argument("--audio_topk", type=int, default=25)
    p.add_argument("--audio_topp", type=float, default=0.95)
    p.add_argument("--audio_repetition_penalty", type=float, default=1.2,
                   help="1.2 is the benchmarked default; 1.0 raises WER.")
    p.add_argument("--seed", type=int, default=None, help="Seed the sampler.")


def _sampling_kwargs(a: argparse.Namespace) -> dict:
    return {
        "cfg_scale": a.cfg_scale,
        "audio_temperature": a.audio_temperature,
        "audio_topk": a.audio_topk,
        "audio_topp": a.audio_topp,
        "audio_repetition_penalty": a.audio_repetition_penalty,
    }


def cmd_say(a: argparse.Namespace) -> int:
    if a.seed is not None:
        np.random.seed(a.seed)
    tts = ZeroTTS.from_pretrained(a.model, revision=a.revision,
                                  intra_op_num_threads=a.threads)

    text = a.text if a.text != "-" else sys.stdin.read()
    # Before chunking: an expansion is several times longer than what it
    # replaces, and the chunk budget has to size the text the model receives.
    if not a.no_text_norm:
        text = normalize_vi_text(text)
    segments = [text]
    if a.chunk:
        segments = [clean_segment_punctuation(s)
                    for s in chunk_text(normalize_punctuation(text),
                                        max_chunk_sec=a.max_chunk_sec)]
        segments = [s for s in segments if s]

    t0 = time.perf_counter()
    chunks = []
    for i, seg in enumerate(segments, 1):
        if len(segments) > 1:
            print(f"[{i}/{len(segments)}] {seg[:70]}{'…' if len(seg) > 70 else ''}",
                  file=sys.stderr)
        chunks.append(tts.synthesize(seg, voice=a.voice, **_sampling_kwargs(a)))
    audio = concat_with_silence(chunks, a.gap_sec, tts.sample_rate)
    elapsed = time.perf_counter() - t0

    tts.save_audio(audio, a.out)
    dur = audio.shape[-1] / tts.sample_rate
    speed = dur / elapsed if elapsed > 0 else float("inf")
    print(f"{a.out}  {dur:.2f}s audio in {elapsed:.2f}s ({speed:.1f}x realtime)")
    return 0


def cmd_voices(a: argparse.Namespace) -> int:
    tts = ZeroTTS.from_pretrained(a.model, revision=a.revision, warmup=False)
    names = tts.list_voices()
    if not names:
        print("No voice packs in this model directory.")
        print("This build cannot create voices from audio — see the README "
              "(voice cloning).")
        return 1
    for name in names:
        v = tts.load_voice(name)
        desc = f"  {v.description}" if v.description else ""
        print(f"{name:20s} {v.language:4s} {v.n_voice_queries} queries{desc}")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    if a.seed is not None:
        np.random.seed(a.seed)
    tts = ZeroTTS.from_pretrained(a.model, revision=a.revision,
                                  intra_op_num_threads=a.threads)
    text = a.text
    timings = []
    for i in range(a.runs):
        timing: dict = {}
        audio = tts.synthesize(text, voice=a.voice, timing=timing,
                               **_sampling_kwargs(a))
        dur = audio.shape[-1] / tts.sample_rate
        timings.append((timing, dur))
        print(f"run {i + 1}: {dur:.2f}s audio, {timing['total_time']:.2f}s wall, "
              f"TTFF {timing['time_to_first_frame'] * 1000:.0f}ms, "
              f"{timing['n_frames']} frames, {dur / timing['total_time']:.1f}x realtime")

    wall = float(np.median([t["total_time"] for t, _ in timings]))
    dur = float(np.median([d for _, d in timings]))
    ttff = float(np.median([t["time_to_first_frame"] for t, _ in timings]))
    print(f"\nmedian over {a.runs}: {dur / wall:.1f}x realtime, TTFF {ttff * 1000:.0f}ms, "
          f"{a.threads} threads")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="zerotts", description="ZeroTTS command line.")
    sub = p.add_subparsers(dest="cmd", required=True)

    say = sub.add_parser("say", help="Synthesize text to a wav file.")
    say.add_argument("text", help="Text to speak, or '-' to read stdin.")
    say.add_argument("-o", "--out", default="out.wav")
    say.add_argument("-v", "--voice", default=None,
                     help="Voice name. Omit for the model's unconditional voice.")
    say.add_argument("--chunk", action="store_true",
                     help="Split long text into segments and join the audio.")
    say.add_argument("--max_chunk_sec", type=float, default=15.0)
    say.add_argument("--gap_sec", type=float, default=0.15,
                     help="Silence inserted between chunks.")
    say.add_argument("--no_text_norm", action="store_true",
                     help="Skip Vietnamese normalization of dates/times/numbers. "
                          "Use for non-Vietnamese text — the expansions are "
                          "Vietnamese words.")
    _add_model_args(say)
    _add_sampling_args(say)
    say.set_defaults(func=cmd_say)

    voices = sub.add_parser("voices", help="List available voices.")
    _add_model_args(voices)
    voices.set_defaults(func=cmd_voices)

    bench = sub.add_parser("bench", help="Measure realtime factor and TTFF.")
    bench.add_argument(
        "--text",
        default="Xin chào, đây là một bài kiểm tra tốc độ tổng hợp giọng nói.")
    bench.add_argument("-v", "--voice", default=None)
    bench.add_argument("--runs", type=int, default=3)
    _add_model_args(bench)
    _add_sampling_args(bench)
    bench.set_defaults(func=cmd_bench)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
