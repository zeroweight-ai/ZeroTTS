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

**The weights are fp32 and not quantized**, so the demo fetches roughly

| | |
|---|---|
| `text_encoder.onnx` | ~280 MB |
| `prefix_step.onnx` | ~391 MB |
| `local_frame_decode.onnx` | ~187 MB |
| codec decoder | ~45 MB |
| **total** | **~900 MB** |

once, then persists it (Cache API / OPFS) so later visits are instant.

Cached copies are keyed by the file's **ETag**, not by its URL. Everything is
fetched from `.../resolve/main/...`, which is a moving target: publishing a new
revision on the Hub leaves the URL identical and the bytes different, so a
URL-keyed cache would serve the old file forever — and silently, since a stale
voice still decodes, it just isn't the voice you shipped. A HEAD per file (six
of them, already needed to size the progress bar) turns a re-published voice
into a 30 KB re-download and leaves the ~900 MB of graphs alone.

This is a deliberate quality-over-size choice, not an oversight. It targets
desktop broadband; it is not suitable for mobile data, and the demo says so
before it starts downloading. If you need a smaller build, quantizing to int8
(~250 MB) or fp16 (~430 MB) is straightforward with
`onnxruntime.quantization` — but validate the result against
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

## Execution providers

WASM, and only WASM. WebGPU is nominally faster, but its kernels are not
bit-identical to the CPU path, and this model's sampling happens *inside* the
graph — small numeric differences change which token is drawn, so the provider
is not a speed knob, it is a change in what the model says. The demo does not
offer it and the loader does not accept it.
