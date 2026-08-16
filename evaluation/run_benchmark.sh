#!/usr/bin/env bash
# Synthesize ZeroBench-TTS with a TTS system, then score it with the benchmark's
# own scorer. See docs/BENCHMARKS.md.
#
#   ./evaluation/run_benchmark.sh
#   MODEL=xtts:capleaf/viXTTS OUT_DIR=./eval/vixtts ./evaluation/run_benchmark.sh
#
#   # keep existing wavs, just re-score them
#   SKIP_EXISTING=1 OUT_DIR=./eval/zerotts ./evaluation/run_benchmark.sh
#
#   # ablation: feed the model the spoken-out text instead of raw orthography
#   SYNTH_FROM=text_normalized OUT_DIR=./eval/zerotts_norm ./evaluation/run_benchmark.sh
#
#   # use a local ZeroBench-TTS clone instead of fetching the scorer
#   ZEROBENCH_DIR=/path/to/ZeroBench-TTS ./evaluation/run_benchmark.sh
#
# Needs `pip install "zerotts[eval]"` — the scorers are torch models; ZeroTTS
# inference itself is not.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCHMARK=${BENCHMARK:-zeroweight-ai/ZeroBench-TTS}
MODEL=${MODEL:-zerotts:zeroweight-ai/ZeroTTS}
OUT_DIR=${OUT_DIR:-./eval/$(echo "${MODEL}" | tr '/:' '__')}

EXTRA=()
[[ -n "${LIMIT:-}" ]]         && EXTRA+=(--limit "$LIMIT")
[[ -n "${SUBSETS:-}" ]]       && EXTRA+=(--subsets $SUBSETS)
[[ -n "${SKIP_EXISTING:-}" ]] && EXTRA+=(--skip_existing)
[[ -n "${GENERATE_ONLY:-}" ]] && EXTRA+=(--generate_only)
[[ -n "${SYNTH_FROM:-}" ]]    && EXTRA+=(--synthesize_from "$SYNTH_FROM")
[[ -n "${ZEROBENCH_DIR:-}" ]] && EXTRA+=(--zerobench_dir "$ZEROBENCH_DIR")
[[ -n "${MODEL_ARGS:-}" ]]    && EXTRA+=($MODEL_ARGS)

python evaluation/run_benchmark.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --out_dir "$OUT_DIR" \
    "${EXTRA[@]}"
