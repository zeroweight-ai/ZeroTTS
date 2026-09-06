# Browser demo

[`js/`](../js/) runs ZeroTTS entirely client-side with
[`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/) — no server, no
upload, no API key. It is a demo in this repository, not a published npm package.

```bash
cd js
npm install
npm run dev        # http://localhost:5173
npm run build      # static bundle in js/dist/, deployable anywhere
```

## The download

Depends on the backend. The default (ggml/GGUF) fetches

| | |
|---|---|
| `gguf/zerotts-f32.gguf` | ~772 MB |
| codec decoder (ONNX) | ~45 MB |
| **total** | **~820 MB** |

and onnxruntime-web fetches the fp32 graphs instead

| | |
|---|---|
| `text_encoder.onnx` | ~323 MB |
| `prefix_step.onnx` | ~348 MB |
| `local_frame_decode.onnx` | ~187 MB |
| codec decoder | ~45 MB |
| **total** | **~903 MB** |

once, then persists it (Cache API / OPFS) so later visits are instant.

(The two ONNX graphs changed size when the cross-attention K/V moved from
`prefix_step` into the text encoder; the total did not.)

Cached copies are keyed by the file's **ETag**, not by its URL. Everything is
fetched from `.../resolve/main/...`, which is a moving target: publishing a new
revision on the Hub leaves the URL identical and the bytes different, so a
URL-keyed cache would serve the old file forever — and silently, since a stale
voice still decodes, it just isn't the voice you shipped. A HEAD per file (six
of them, already needed to size the progress bar) turns a re-published voice
into a 30 KB re-download and leaves the weights alone.

The ONNX path is fp32 by choice: int8 quantization of these graphs was tried
and came out **slower** than fp32 in WebAssembly, for the same instruction-set
reason described under the ggml backend below. If you want a smaller download,
take the GGUF backend rather than quantizing these. Either way, validate against
[the benchmark](BENCHMARKS.md) before trusting it, because nothing here has been
measured at reduced precision.

`text_encoder.onnx` runs **once per utterance**, not per frame, so it can be
loaded lazily after the two hot-path graphs and does not delay the first audio.

## Structure

```
js/src/
  synthesizer.ts   the two-calls-per-frame loop (see docs/RUNTIME.md)
  codec.ts         MOSS decoder: batch + KV-cached streaming
  tokenizer.ts     BPE over tokenizer.json
  chunking.ts      long-form segmentation (port of zerotts.chunking)
  loader.ts        create sessions from the downloaded graphs
  repo.ts          resolve repo URLs, size the download, load voices
  worker.ts        the Web Worker the model runs in
  workerClient.ts  main-thread handle on that worker
  cache.ts         Cache API persistence, download progress
  rng.ts           seedable PRNG — the sampler's draws are graph inputs
  player.ts        AudioWorklet ring buffer for streaming playback
  samples.ts       sample texts, shared with the Python UI
  main.ts          demo UI wiring (imports no runtime code)
```

Audio is pushed into an `AudioWorklet` ring buffer rather than scheduled as
individual `AudioBufferSourceNode`s, so chunk boundaries don't click.

Generation runs in a **Web Worker** (`js/src/worker.ts`). ORT-web's WASM backend
computes on whatever thread calls it, and this model does two graph calls per
80 ms frame for up to 1500 frames — on the UI thread that blocks painting, input
and even the AudioWorklet's message port for the whole take, so the tab appears
frozen. The page therefore imports no runtime code at all: it sends text and
receives decoded `Float32Array` chunks.

Two consequences worth knowing:

- Cancellation must be a *macrotask* away. `await session.run()` resolves in a
  microtask after synchronous compute, so a generation loop that only awaits ORT
  never drains the worker's message queue and a `cancel` message would sit unread
  until the take finished. The worker yields the event loop once per decoded
  chunk (via a `MessageChannel`, since timers in a hidden tab's worker are
  clamped), which puts Stop's latency at well under a second.
- Chunks are *transferred*, not copied — the codec allocates a fresh array per
  chunk and never reads it again.

## Porting notes

[docs/RUNTIME.md](RUNTIME.md) is the contract; these are the things that
specifically bite in JavaScript.

* **Two integer widths.** The TTS graphs take `int64` (`BigInt64Array` in
  ORT-web); the codec graphs take `int32`. Mixing them fails at session run with
  a type error.
* **`voice.bin`, not `voice.npz`.** `.npz` is a zip of `.npy`; the browser gets a
  raw little-endian float32 blob instead — `fetch` → `arrayBuffer` →
  `new Float32Array`.
* **`seen_mask` is `(1, K, 1024)` bool**, mutated in place every frame. Use a
  `Uint8Array` and wrap it as `new ort.Tensor('bool', buf, dims)`; do not
  reallocate it per frame.
* **`packed_kv` grows every frame.** Reallocating a few hundred MB per frame will
  dominate the runtime — preallocate to `max_frames` and slice.
* **Seed the RNG yourself.** `ctrl_random_u` / `audio_random_u` are graph inputs,
  so a seeded PRNG makes the JS port bit-comparable against the Python runtime.
  That is the parity fixture: same text, same voice, same draws → identical frame
  codes. It is the only practical way to keep a port correct across re-exports.
* **External data files.** The `.data` files beside the codec graphs must be
  registered with ORT-web explicitly; they are not fetched implicitly.

## A second backend: ggml/GGUF — now the default

[`cpp/`](../cpp/) reimplements the generation loop on ggml, compiled to WASM,
reading a GGUF file instead of the three ONNX graphs. The demo uses it by
default in its **unquantized** form: 6.1x realtime against onnxruntime's 4.2x,
bit-exact against it, and a 772 MB download against ~858 MB. Quantized GGUFs
(206 MB / 124 MB) are selectable when download size is the binding constraint.
onnxruntime-web remains selectable, and is still the only backend that
implements CFG.

Two findings from building it are worth keeping:

**Quantization does not buy speed in WebAssembly.** Quantized weights are
*slower* than fp32 at every thread count measured — SIMD128 has no integer
dot-product instruction, so a q8_0 block costs ~20 SIMD ops per 32 MACs against
fp32's 8 FMAs. (The same instruction-set gap is why quantizing the ONNX graphs
to int8 also came out slower.) fp32 GGUF is the fastest build at every thread
count, which is why it is the default despite being the largest — quantization
here buys download size and costs both speed and exactness.

**The largest single win was algorithmic, and it has been folded back into the
ONNX graphs.** `prefix_step.onnx` took `text_states` as a per-frame input, so it
re-projected every decoder layer's cross-attention K/V over the whole text on
each frame — 45 us per text token per frame, for a result that cannot change
within an utterance. The text encoder now emits those K/V once and `prefix_step`
consumes them: byte-identical output, 1.07x on a short sentence and **1.28x on
the ~120-token segments `chunkText` produces**, and per-frame cost no longer
depends on text length. That closed most of the gap between the two backends.

See [cpp/README.md](../cpp/README.md) for the measurements and
[js/bench-ggml.html](../js/bench-ggml.html) (under `npm run dev`) to re-run them
on other hardware — worth doing, since the fp32-beats-quantized result should
invert on a machine that is short of memory bandwidth rather than of compute.

## Execution providers

WASM, and only WASM. WebGPU is nominally faster, but its kernels are not
bit-identical to the CPU path, and this model's sampling happens *inside* the
graph — small numeric differences change which token is drawn, so the provider
is not a speed knob, it is a change in what the model says. The demo does not
offer it and the loader does not accept it.
