#!/usr/bin/env bash
# Run a TTS system over ZeroBench-TTS. See docs/BENCHMARKS.md.
#
#   ./evaluation/run_benchmark.sh
#   MODEL=xtts:capleaf/viXTTS OUT_DIR=./eval/vixtts ./evaluation/run_benchmark.sh
#   USE_VINORM=1 SKIP_EXISTING=1 OUT_DIR=./eval/zerotts_vinorm ./evaluation/run_benchmark.sh
#
# Needs `pip install "zerotts[eval]"` (the scorers are torch models; ZeroTTS
# inference itself is not).
set -euo pipefail
cd "$(dirname "$0")/.."

BENCHMARK=${BENCHMARK:-zeroweight-ai/ZeroBench-TTS}
MODEL=${MODEL:-zerotts:zeroweight-ai/ZeroTTS}
OUT_DIR=${OUT_DIR:-./eval/$(echo "${MODEL}" | tr '/:' '__')}
ASR=${ASR:-vinai/PhoWhisper-large}

EXTRA=()
[[ -n "${LIMIT:-}" ]]         && EXTRA+=(--limit "$LIMIT")
[[ -n "${SUBSETS:-}" ]]       && EXTRA+=(--subsets $SUBSETS)
[[ -n "${USE_VINORM:-}" ]]    && EXTRA+=(--use_vinorm)
[[ -n "${SKIP_EXISTING:-}" ]] && EXTRA+=(--skip_existing)
[[ -n "${MODEL_ARGS:-}" ]]    && EXTRA+=($MODEL_ARGS)

python evaluation/run_benchmark.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --out_dir "$OUT_DIR" \
    --asr_model "$ASR" \
    "${EXTRA[@]}"
