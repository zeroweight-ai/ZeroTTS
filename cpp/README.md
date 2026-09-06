# ZeroTTS on ggml — a GGUF runtime for the browser

A second implementation of ZeroTTS's generation loop, built directly on
[ggml](https://github.com/ggml-org/ggml) (llama.cpp's tensor library) and
compiled to WASM, as an alternative to the `onnxruntime-web` path in
[`../js/`](../js/). It was built to answer one question — **does a quantized
GGUF make the browser demo faster?** — and the answer turned out to be more
interesting than yes or no.

**Short version:** two independent findings.

1. **Quantization does not make this faster in WebAssembly.** Quantized weights
   are *slower* than fp32 at every thread count measured. Their payoff is a
   4-7x smaller download at roughly break-even speed.
2. **The big win was an algorithmic one, and it did not need this port.**
   `prefix_step.onnx` was re-projecting the text's cross-attention K/V on every
   frame; precomputing them once per utterance is worth ~1.3x on realistic
   segment lengths. That fix has since been applied to the ONNX graphs, which
   is where most of this port's original margin went.

What is left after that: the unquantized ggml path is ~1.5x faster than
onnxruntime-web at each backend's shipping thread count, and slightly smaller on
the wire. Choosing a quantized GGUF trades that speed away for download size.

## Results

Chrome, 14-core Apple silicon, cross-origin isolated, one 3.2 s utterance
(40 frames), frame generation only — no codec, no audio. Higher is better. Each
figure is the median of repeated runs in a freshly loaded page; see the note at
the end of this section about why that matters.

**At the defaults each backend actually ships with:**

| backend | threads | download | realtime |
|---|---:|---:|---:|
| onnxruntime-web (with the cross-K/V fix) | 4 | ~858 MB | 4.2x |
| **ggml f32** | 8 | **772 MB** | **6.2x** |

The demo defaults to the **unquantized** GGUF. That reads backwards until you
look at the table below: f32 is the fastest build at every thread count *and*
the only one that reproduces the ONNX frame codes exactly, and at 772 MB it is
still a smaller download than the graphs it replaces. The quantized builds are
there for when download size is the binding constraint, and they cost both
speed and exactness to get it.

onnxruntime-web caps itself at 4 threads (more has never helped it here); the
ggml backend takes `min(hardwareConcurrency, 8)`.

**ggml across quantization and thread count** (onnxruntime is 4.2x at its 4):

| threads | ggml f32 | ggml f16 | ggml q8_0 | ggml q5_0 | ggml q4_0 |
|--------:|---------:|---------:|----------:|----------:|----------:|
| 1 | **2.40x** | 0.41x | 1.06x | 0.59x | 1.27x |
| 2 | **3.98x** | 0.79x | 1.99x | 1.12x | 2.35x |
| 4 | **6.1x** | 1.49x | 3.6x | 2.06x | 4.11x |
| 8 | **6.2x** | 2.57x | 5.3x | 3.42x | 5.8x |

Weights on the wire, and how far each drifts from fp32:

| build | size | codes drawn differently vs f32 |
|---|---:|---:|
| ONNX graphs | ~858 MB | 0 (bit-exact — same weights) |
| GGUF f32 | 772 MB | 0 (bit-exact) |
| GGUF f16 | 405 MB | 0.4% |
| GGUF q8_0 | 206 MB | 4.9% |
| GGUF q5_0 | 145 MB | 18.9% |
| GGUF q4_0 | 124 MB | 36.5% |

The drift column is measured teacher-forced (`zerotts-quality`, below), not by
comparing two independent takes: two models sampling on their own diverge at the
first differing token and every later frame is then a different utterance, which
measures nothing. The control channel — the `<eoa>` stop decision — was
identical in every build.

> **Measuring this yourself:** hold only one backend in the page at a time. Each
> is 0.2-0.9 GB of buffers plus a WASM heap, and keeping two alive pushes the
> renderer into swapping. That does not fail — it makes every later run 10-20x
> slower, uniformly, and reads exactly like a real regression. `bench-ggml.html`
> releases the previous backend before loading the next for this reason; two
> hours of the numbers above were thrown away before that was understood.

### Why quantization loses here

WASM's SIMD128 has no integer dot-product instruction. ggml's WASM kernel for a
q8_0 block spends roughly twenty SIMD ops (four `i16x8` extends per operand, four
`i32x4_dot_i16x8`, the adds, plus a scalar fp16 scale conversion) on 32 MACs,
where the fp32 path issues eight `f32x4` FMAs for the same 32 MACs — and it must
quantize the activation vector first. Batch-1 decoding *is* bandwidth-bound in
principle, but on hardware with bandwidth to spare the extra instructions cost
more than the saved bytes. f16 is worse still: there is no f16 SIMD at all, so
every weight goes through a conversion.

This is the same reason the earlier int8 ONNX experiment came out slower than
fp32. It is a property of the WASM instruction set, not of onnxruntime or of
ggml, and no amount of re-quantizing will change it.

The picture should invert on a genuinely bandwidth-starved machine, and the
thread scaling above hints at it — quantized builds gain much more from extra
threads than fp32 does, because they are the ones still short of compute.
[`../js/bench-ggml.html`](../js/bench-ggml.html) runs this whole matrix in one
page (`cd ../js && npm run dev`, then open `/bench-ggml.html` — it needs the dev
server's COOP/COEP headers for threads, and reads the GGUF files from
`js/public/ggml/models`, which is a symlink to `cpp/models`). It is worth
re-running on the Windows machines that are below realtime rather than assuming
these numbers transfer.

### Where the speedup came from, and what happened to it

Two separate things, and only one of them needed a new runtime. Timed per stage
against the **original** ONNX graphs, fp32 both sides:

| per frame | onnxruntime (before) | ggml | |
|---|---:|---:|---|
| global decoder step | 6.27 ms | 2.17 ms | **2.9x** |
| local frame decode | 12.01 ms | 10.95 ms | 1.1x |

The decoder-step gap had an L-dependent half. `prefix_step.onnx` took
`text_states` as an input on **every** frame, so it re-ran each decoder layer's
cross-attention K and V projections over the whole text every time — for a
result that cannot change within an utterance. This port projects them once in
`zerotts_begin`, which is why its decoder step barely moves with text length
while the old ONNX one climbed:

| text tokens | onnx `prefix_step` (before) | ggml advance |
|---:|---:|---:|
| 31 | 6.27 ms | 2.17 ms |
| 61 | 7.89 ms | 2.23 ms |
| 91 | 9.25 ms | 2.34 ms |
| 121 | 10.36 ms | 2.38 ms |

45.4 us per token per frame against 2.3 us. The residual slope is the
cross-attention *scores*, which are genuinely per-frame work.

A naive flop count overstates this: the cross-K/V projection is a
matrix-*matrix* product running near peak, while the rest of the step is
batch-1 mat-*vec* running at a fraction of it, so time per MAC differs by an
order of magnitude between them. Measured, the recompute was about a third of
the 2.9x at 31 tokens and the majority only past ~65.

**The ONNX graphs now do this too.** `inference/export_onnx.py` emits the
per-layer cross K/V from the text encoder and `prefix_step` consumes them, which
was verified to produce byte-identical frame codes and measured at:

| text tokens | before | after | speedup |
|---:|---:|---:|---:|
| 31 | 4.39x realtime | 4.71x | 1.07x |
| 121 | 3.55x | 4.53x | **1.28x** |

The demo chunks at 225 characters (`chunkText`'s 15 s x 15 chars/s), which is
~120 tokens, so real segments sit on the bottom row. The flat right-hand column
is the point: a longer segment used to cost more per frame and now does not.

That change is most of why the gap between the two backends narrowed from
~1.8x to ~1.25x. What remains is ggml's batch-1 kernels and the fact that it
scales past four threads, where onnxruntime-web does not.

## Layout

| path | role |
|---|---|
| `include/zerotts.h` | the C API — the call sequence mirrors [`docs/RUNTIME.md`](../docs/RUNTIME.md) |
| `src/zerotts.cpp` | model loading and the four ggml graphs |
| `src/wasm.cpp` | Emscripten entry points |
| `convert/convert_zerotts_gguf.py` | checkpoint → GGUF |
| `test/bench.cpp` | native driver; prints frame codes and per-stage timings |
| `test/quant_quality.cpp` | teacher-forced quantization drift |
| `test/wasm_bench.mjs` | drives the WASM build from Node — thread sweeps without a browser |
| `../js/src/ggmlBackend.ts` | the browser-side wrapper |
| `../js/test/ggml-parity.mjs` | ggml vs ONNX frame codes |
| `../js/bench-ggml.html` | the A/B page the table above came from (dev server only) |

Four graphs, not three: the text encoder and the cross-K/V precompute run
together once per segment, the voice block runs as its own pass, and then the
`<soa>` token and every audio frame share one decoder-step graph. Splitting the
prefix that way is what lets the whole runtime avoid building an attention mask
anywhere — see the comment on `zt_run_voice_prefix`.

## Correctness

The f32 build is **bit-exact against the ONNX runtime**: same text, voice and
draw sequence, identical frame codes, verified natively and through the actual
WASM artifact the browser loads. Both sides take the sampler's uniforms as
inputs, which is what makes an exact comparison possible at all.

```bash
cd ../js
npm install --no-save onnxruntime-node
node --loader ./test/ts-loader.mjs test/ggml-parity.mjs \
     dist/model ../cpp/models/zerotts-f32.gguf
```

A quantized GGUF is *expected* to diverge — it samples from slightly different
logits — so the script reports the difference and only fails on f32.

Two things this port had to get right that fail silently if you don't:

* **Raw weights, not the EMA shadow.** `export_compact.py` in the research repo
  merges EMA by default, but the shipped ONNX graphs were exported without it
  (their LayerNorm initializers match the raw weights exactly and the EMA ones to
  ~1e-4). Converting with `--ema` yields a model that is bit-exact against
  nothing and drifts ~5% in the global hidden state — a demo whose two backends
  quietly say different things.
* **RoPE pair ordering.** `GGML_ROPE_TYPE_NORMAL` rotates adjacent pairs, which
  matches `transformer_block._apply_rope`'s `x.view(..., half, 2)`. NEOX's
  half-and-half ordering runs fine and degrades the output.

## Building

```bash
git submodule update --init cpp/vendor-ggml

# native — for the parity and quality tests
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build -j

# browser — needs the Emscripten SDK on PATH
source ~/emsdk/emsdk_env.sh
./build-wasm.sh          # writes ../js/public/ggml/
```

Weights:

```bash
conda run -n tts python convert/convert_zerotts_gguf.py \
    /path/to/checkpoint.pt -o models/zerotts-q8_0.gguf --quant q8_0
```

`--quant` takes `f32`, `f16`, `q8_0`, `q5_0` or `q4_0`; `--embed-quant` sets a
separate (usually higher) precision for the embedding tables and output heads,
which are read once per depth step and so buy little bandwidth for what
quantizing them costs.

Then:

```bash
./build/zerotts-bench   models/zerotts-q8_0.gguf voice.bin ids.txt [seed] [threads]
./build/zerotts-quality models/zerotts-f32.gguf models/zerotts-q4_0.gguf voice.bin ids.txt
```

`ids.txt` is whitespace-separated BPE ids, so both runtimes are fed exactly the
same tokens and any difference is the model rather than the text front end.

## Not implemented

* **CFG.** `cfgScale > 1` needs a batch-2 decoder; the demo defaults it off and
  the backend throws rather than silently ignoring it.
* **The codec.** Waveform decoding still runs on onnxruntime — it is ~45 MB and
  not on the per-frame hot path.
* **The voice encoder.** The browser gets precomputed `voice.bin` latents, so
  the encoder's 22 M parameters are not converted at all.
