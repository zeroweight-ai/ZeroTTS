"""Synthesize ZeroBench-TTS with any registered TTS backend, then score it.

    python evaluation/run_benchmark.py \\
        --benchmark zeroweight-ai/ZeroBench-TTS \\
        --model zerotts:zeroweight-ai/ZeroTTS \\
        --out_dir ./eval/zerotts

This script **only synthesizes**. Every metric — the two-ASR WER, the
acceptable-reference expansion, SSIM, UTMOS, silence — comes from
``zerobench_eval``, the official scorer published inside the `ZeroBench-TTS
<https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS>`_ dataset repo and
fetched automatically (see ``evaluation/zerobench.py``).

That split is deliberate. A scorer vendored into the repo of the model being
scored can drift from the benchmark's — silently, and always in the flattering
direction. Here the same code scores ZeroTTS and everything it is compared to,
and you can run it yourself against any system without touching this repo:

    python -m zerobench_eval score --wav_dir my_wavs/ --name MyModel

Adding a backend is a subclass in ``evaluation/tts_models.py``; this runner only
ever calls ``synthesize(TTSRequest) -> np.ndarray``.

Outputs: {out_dir}/wav/{subset}/{item_id}.wav, plus the scorer's per_sample.csv,
summary.json and report.txt. See docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.tts_models import TTSRequest, build_model, parse_model_args  # noqa: E402
from evaluation.zerobench import ensure_zerobench  # noqa: E402

SUBSETS = ("vietnamese", "code_switch", "cross_lingual", "challenging")


def _log(msg: str) -> None:
    print(f"[benchmark] {msg}", flush=True)


def load_benchmark(path: str, subsets, limit) -> tuple:
    """Read the benchmark. ``path`` is a local build dir, a metadata.jsonl, or an
    HF dataset repo id (parquet configs are pulled and audio re-materialized as
    plain wavs so backends still get file paths)."""
    p = Path(path)
    if p.is_dir() and (p / "metadata.jsonl").exists():
        rows = [json.loads(x) for x in (p / "metadata.jsonl").read_text().splitlines() if x.strip()]
        root = p
    elif p.suffix == ".jsonl" and p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        root = p.parent
    else:
        rows, root = _load_from_hub(path, subsets)

    for r in rows:
        r.setdefault("item_id", r.get("voice_id"))
    if subsets:
        rows = [r for r in rows if r.get("subset") in set(subsets)]
    rows.sort(key=lambda r: (r.get("subset", ""), str(r.get("item_id"))))
    if limit:
        kept, seen = [], {}
        for r in rows:
            s = r.get("subset", "")
            if seen.get(s, 0) < limit:
                kept.append(r)
                seen[s] = seen.get(s, 0) + 1
        rows = kept
    return rows, root


def _load_from_hub(repo_id: str, subsets) -> tuple:
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    cache = Path.home() / ".cache" / "zerobench_tts" / repo_id.replace("/", "__")
    audio_dir = cache / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in (subsets or SUBSETS):
        try:
            ds = load_dataset(repo_id, config, split="test")
        except Exception as exc:
            _log(f"skipping config {config}: {exc}")
            continue
        ds = ds.cast_column("ref_audio", Audio(decode=False))
        for row in ds:
            item_id = str(row.get("item_id") or row.get("voice_id"))
            wav_path = audio_dir / f"{config}_{item_id}.wav"
            if not wav_path.exists():
                data, sr = sf.read(io.BytesIO(row["ref_audio"]["bytes"]))
                sf.write(wav_path, data, sr)
            rows.append({**{k: v for k, v in row.items() if k != "ref_audio"},
                         "subset": config, "item_id": item_id,
                         "ref_audio": str(wav_path)})
    return rows, cache


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="zeroweight-ai/ZeroBench-TTS")
    ap.add_argument("--model", required=True, help="<backend>[:<path-or-repo>]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_arg", action="append", default=[])
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Per subset.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Keep wavs already in out_dir instead of regenerating.")
    ap.add_argument("--synthesize_from", choices=("text", "text_normalized"), default="text",
                    help="Which field to FEED THE MODEL. 'text' (default) is the "
                         "benchmark task: raw orthography, digits and acronyms "
                         "included. 'text_normalized' feeds the spoken-out form "
                         "instead — an ablation simulating a perfect Vietnamese "
                         "text-normalization frontend. Scoring references are "
                         "unchanged either way, so the two are comparable.")
    ap.add_argument("--generate_only", action="store_true",
                    help="Synthesize and stop; score later with "
                         "`python -m zerobench_eval score --wav_dir <out_dir>/wav`.")
    ap.add_argument("--zerobench_dir", default=None,
                    help="Local ZeroBench-TTS clone providing zerobench_eval/ "
                         "(default: $ZEROBENCH_DIR, else fetched from the Hub).")
    ap.add_argument("--skip_utmos", action="store_true",
                    help="Skip UTMOSv2 during scoring (optional dependency).")
    args = ap.parse_args()

    rows, bench_root = load_benchmark(args.benchmark, args.subsets, args.limit)
    if not rows:
        raise SystemExit(f"no benchmark items matched (subsets={args.subsets})")
    _log(f"{len(rows)} items from {args.benchmark}")

    # Resolve the scorer BEFORE synthesizing, so a missing checkout or a broken
    # network fails in seconds rather than after an hour of generation.
    if not args.generate_only:
        ensure_zerobench(args.zerobench_dir)

    model = build_model(args.model, device=args.device, **parse_model_args(args.model_arg))
    _log(f"model '{model.name}' ready, sample_rate={model.sample_rate}")
    if args.synthesize_from != "text":
        _log(f"NOTE synthesizing from '{args.synthesize_from}' — this is the "
             "text-normalization ablation, NOT the benchmark task.")

    import soundfile as sf

    out_dir = Path(args.out_dir)
    wav_root = out_dir / "wav"
    n_new = n_empty = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        wav_path = wav_root / row["subset"] / f"{row['item_id']}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and wav_path.exists():
            continue

        # Only the model's INPUT switches; scoring always uses the benchmark's
        # own text/text_normalized, so both modes stay directly comparable.
        text = row.get(args.synthesize_from) or row["text"]
        audio = model.synthesize(TTSRequest(
            text=text, ref_audio=row["ref_audio"], language=row.get("lang", "vi"),
            subset=row["subset"], item_id=row["item_id"],
        ))
        if audio is None or audio.size == 0:
            n_empty += 1
            _log(f"  [{row['item_id']}] WARNING empty generation")
            continue
        sf.write(str(wav_path), audio, model.sample_rate)
        n_new += 1
        if i % 10 == 0 or i == len(rows):
            _log(f"  synthesized {i}/{len(rows)}  [{time.time() - t0:.0f}s]")

    _log(f"{n_new} wavs written to {wav_root} ({n_empty} empty generations)")

    if args.generate_only:
        _log("--generate_only: skipping scoring. Score with:")
        _log(f"    python -m zerobench_eval score --wav_dir {wav_root} --name {model.name}")
        return

    # ── hand off to the benchmark's own scorer ────────────────────────────────
    from zerobench_eval.__main__ import cmd_score

    _log("scoring with zerobench_eval (the benchmark's official scorer)")
    # Only hand over a benchmark path that actually holds metadata.jsonl. When
    # the rows came from the Hub, bench_root is an audio-only cache dir, so pass
    # None and let the scorer resolve the benchmark itself.
    local_meta = Path(str(bench_root)) / "metadata.jsonl"
    cmd_score(argparse.Namespace(
        benchmark=str(bench_root) if local_meta.exists() else None,
        wav_dir=str(wav_root), name=model.name, out_dir=str(out_dir),
        subsets=args.subsets, device=args.device, asr=None,
        skip_utmos=args.skip_utmos, allow_missing=bool(args.limit),
    ))

    (out_dir / "generation.json").write_text(json.dumps({
        "model": model.name, "model_spec": args.model, "model_config": model.config(),
        "benchmark": args.benchmark, "synthesize_from": args.synthesize_from,
        "n_items": len(rows), "n_empty_generations": n_empty,
        "wall_clock_sec": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwavs -> {wav_root}/<subset>/   metrics -> {out_dir}/per_sample.csv")


if __name__ == "__main__":
    main()
