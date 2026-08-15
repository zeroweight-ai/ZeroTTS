"""Run any registered TTS backend over ZeroBench-TTS and report WER/SSIM/UTMOS.

    python evaluation/run_benchmark.py \
        --benchmark zeroweight-ai/ZeroBench-TTS \
        --model zerotts:zeroweight-ai/ZeroTTS \
        --out_dir ./eval/zerotts_asis

    # re-score existing wavs with the vinorm reference — no re-synthesis
    python evaluation/run_benchmark.py ... --use_vinorm --skip_existing \
        --out_dir ./eval/zerotts_vinorm

Outputs {out_dir}/wav/{subset}/{item_id}.wav, per_sample.csv, summary.json, and
a printed report. See docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.scorers import (  # noqa: E402
    DEFAULT_ASR,
    ASRScorer,
    SSIMScorer,
    UTMOSScorer,
    excess_silence,
    resample_to_16k,
    score_wer,
)
from evaluation.tts_models import TTSRequest, build_model, parse_model_args  # noqa: E402

SUBSETS = ("vietnamese", "code_switch", "cross_lingual", "challenging")

_vinorm = None


def _log(msg: str) -> None:
    print(f"[benchmark] {msg}", flush=True)


def wer_references(row: dict, use_vinorm: bool) -> list:
    """Accepted transcriptions for this item.

    Always the written text, plus the benchmark's hand-curated spoken-out
    normalization when it differs. ``use_vinorm`` adds a third from soe-vinorm's
    automatic normalization — see docs/BENCHMARKS.md for why that is a separate
    reported column rather than always on.
    """
    refs = [row["text"]]
    normalized = row.get("text_normalized")
    if normalized and normalized != row["text"]:
        refs.append(normalized)
    if use_vinorm:
        global _vinorm
        if _vinorm is None:
            from soe_vinorm import normalize_text

            _vinorm = normalize_text
        try:
            auto = _vinorm(row["text"])
            if auto and auto not in refs:
                refs.append(auto)
        except Exception as exc:
            _log(f"vinorm failed on {row.get('item_id')}: {exc}")
    return refs


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

    if subsets:
        rows = [r for r in rows if r.get("subset") in set(subsets)]
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
    from datasets import load_dataset

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
        from datasets import Audio

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


def aggregate(rows: list) -> dict:
    def stat(key, fn):
        vals = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
        return float(fn(vals)) if vals else float("nan")

    out = {"n": len(rows)}
    for key in ("wer", "ssim", "utmos", "excess_silence"):
        out[f"{key}_mean"] = stat(key, statistics.mean)
        out[f"{key}_median"] = stat(key, statistics.median)
    return out


def format_report(title: str, groups: dict) -> str:
    lines = [f"\n{title}", "=" * len(title),
             f"{'group':<22}{'n':>5}{'WER':>10}{'WERmed':>10}{'SSIM':>9}"
             f"{'UTMOS':>8}{'SIL(s)':>9}"]
    for name, agg in groups.items():
        lines.append(
            f"{name:<22}{agg['n']:>5}{agg['wer_mean'] * 100:>9.2f}%"
            f"{agg['wer_median'] * 100:>9.2f}%{agg['ssim_mean']:>9.3f}"
            f"{agg['utmos_mean']:>8.2f}{agg['excess_silence_mean']:>9.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="zeroweight-ai/ZeroBench-TTS")
    ap.add_argument("--model", required=True, help="<backend>[:<path-or-repo>]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_arg", action="append", default=[])
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Per subset.")
    ap.add_argument("--asr_model", default=DEFAULT_ASR)
    ap.add_argument("--use_vinorm", action="store_true",
                    help="Add soe-vinorm's automatic normalization as a WER reference.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Re-score wavs already in out_dir instead of regenerating.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)

    rows, _root = load_benchmark(args.benchmark, args.subsets, args.limit)
    _log(f"{len(rows)} items from {args.benchmark}")

    import soundfile as sf

    model = None
    if not args.skip_existing:
        model = build_model(args.model, device=args.device, **parse_model_args(args.model_arg))
        _log(f"model: {model.name}")

    _log(f"scorers: ASR={args.asr_model}, vinorm={'on' if args.use_vinorm else 'off'}")
    asr = ASRScorer(args.asr_model, device=args.device)
    ssim = SSIMScorer(device=args.device)
    utmos = UTMOSScorer(device=args.device)

    results = []
    n_empty = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        subset = row.get("subset", "")
        item_id = str(row.get("item_id"))
        wav_path = out_dir / "wav" / subset / f"{item_id}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and wav_path.exists():
            audio, sr = sf.read(wav_path)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            req = TTSRequest(text=row["text"], ref_audio=row["ref_audio"],
                             language=row.get("lang", "vi"), subset=subset,
                             item_id=item_id,
                             extra=dict(row))
            audio = model.synthesize(req)
            sr = model.sample_rate
            if audio.size == 0:
                n_empty += 1
                _log(f"[{i}/{len(rows)}] {item_id}: EMPTY generation")
                continue
            sf.write(wav_path, audio, sr, subtype="PCM_16")

        gen16 = resample_to_16k(audio, sr)
        ref_audio, ref_sr = sf.read(row["ref_audio"])
        ref16 = resample_to_16k(np.asarray(ref_audio, dtype=np.float32).reshape(-1), ref_sr)

        transcript = asr.transcribe(gen16, lang=row.get("lang", "vi"))
        wer, ref_idx = score_wer(transcript, wer_references(row, args.use_vinorm))

        results.append({
            "item_id": item_id, "subset": subset,
            "length_bucket": row.get("length_bucket", ""),
            "voice_source": row.get("voice_source", ""),
            "text": row["text"], "transcript": transcript,
            "wer": wer, "wer_ref_index": ref_idx,
            "ssim": ssim.score(gen16, ref16),
            "utmos": utmos.score(str(wav_path)),
            "excess_silence": excess_silence(audio, sr),
            "duration_sec": len(audio) / sr,
        })
        if i % 10 == 0 or i == len(rows):
            _log(f"[{i}/{len(rows)}] running WER "
                 f"{np.mean([r['wer'] for r in results]) * 100:.2f}%")

    with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    def group_by(key):
        groups = {}
        for value in sorted({r[key] for r in results if r.get(key)}):
            groups[value] = aggregate([r for r in results if r.get(key) == value])
        return groups

    summary = {
        "model": args.model, "benchmark": args.benchmark,
        "asr_model": args.asr_model, "use_vinorm": args.use_vinorm,
        "model_config": model.config() if model else None,
        "n_items": len(rows), "n_scored": len(results),
        "n_empty_generations": n_empty,
        "wall_clock_sec": round(time.perf_counter() - t0, 1),
        "overall": aggregate(results),
        "by_subset": group_by("subset"),
        "by_length_bucket": group_by("length_bucket"),
        "by_voice_source": group_by("voice_source"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(format_report(f"{args.model} — overall", {"overall": summary["overall"]}))
    print(format_report("by subset", summary["by_subset"]))
    if n_empty:
        print(f"\nWARNING: {n_empty} empty generation(s)")
    _log(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
