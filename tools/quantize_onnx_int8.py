#!/usr/bin/env python3
"""Quantize the ZeroTTS ONNX graphs to int8 (dynamic, MatMul-only).

Why this exists: the fp32 graphs are memory-bandwidth-bound on an ordinary
DDR4 desktop, not compute-bound. ``local_frame_decode.onnx`` unrolls the
4-layer local transformer 16x (once per codebook) into ~7,500 nodes, and every
depth step re-reads all 34.6 M weights — ~2.2 GB of weight traffic per frame,
enough to saturate DDR4. int8 quarter-cuts that traffic, turning RTF from >1x
into ~0.5x on such a machine (see docs/BENCHMARKS.md).

The trap: a plain ``quantize_dynamic(op_types_to_quantize=["MatMul"])`` only
quantizes 41 of the 561 MatMuls in ``local_frame_decode``, because the weights
reach the MatMuls through ``Identity`` nodes the quantizer does not look
through. Fix: let ORT fold the graph first (``ORT_ENABLE_BASIC`` +
``optimized_model_filepath``), which removes the Identities, then quantize —
425 ``MatMulInteger`` nodes.

Usage:
    python tools/quantize_onnx_int8.py --src models/zerotts --dst models/zerotts-int8

Everything is copied verbatim (config, tokenizer, voices, codec) and the three
hot graphs are replaced with their int8 versions. The codec decoder stays fp32 —
it is not on the per-frame hot path.
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import time
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

# The three hot graphs (docs/RUNTIME.md). The codec decoder under onnx/codec/
# is copied through untouched.
GRAPHS = ("text_encoder.onnx", "prefix_step.onnx", "local_frame_decode.onnx")


def quantize_one(src: Path, dst: Path) -> dict:
    """Fold Identity nodes, then dynamic-quantize the MatMuls. Returns stats."""
    folded = dst.with_name(dst.name + ".opt")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    # Session creation runs the basic optimizations and writes the folded graph
    # to ``optimized_model_filepath``; the session is never actually run.
    ort.InferenceSession(str(src), options, providers=["CPUExecutionProvider"])

    started = time.perf_counter()
    try:
        quantize_dynamic(
            str(folded),
            str(dst),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
    finally:
        folded.unlink(missing_ok=True)

    counts = collections.Counter(
        n.op_type for n in onnx.load(str(dst), load_external_data=False).graph.node
    )
    return {
        "size_in": src.stat().st_size,
        "size_out": dst.stat().st_size,
        "matmul": counts["MatMul"],
        "matmul_integer": counts["MatMulInteger"],
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, default=Path("models/zerotts"),
                    help="fp32 model directory (default: models/zerotts)")
    ap.add_argument("--dst", type=Path, default=Path("models/zerotts-int8"),
                    help="int8 output directory (default: models/zerotts-int8)")
    args = ap.parse_args()

    if not (args.src / "config.json").is_file():
        ap.error(f"{args.src} does not look like a ZeroTTS model directory "
                 "(no config.json)")

    shutil.copytree(args.src, args.dst, dirs_exist_ok=True)

    total_in = total_out = 0
    print(f"{'graph':28s} {'fp32':>7s} {'int8':>7s} {'MatMul':>7s} "
          f"{'MatMulInteger':>13s}   time")
    for name in GRAPHS:
        stats = quantize_one(args.src / "onnx" / name, args.dst / "onnx" / name)
        total_in += stats["size_in"]
        total_out += stats["size_out"]
        print(f"{name:28s} {stats['size_in'] / 1e6:7.0f}M "
              f"{stats['size_out'] / 1e6:7.0f}M {stats['matmul']:7d} "
              f"{stats['matmul_integer']:13d} {stats['seconds']:5.1f}s")

    print(f"{'total':28s} {total_in / 1e6:7.0f}M {total_out / 1e6:7.0f}M "
          f"  ({total_in / total_out:.2f}x smaller)")
    print(f"\nwrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
