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
  voices.ts        fetch voices/index.json + voice.bin
  cache.ts         OPFS persistence, download progress
  worker.ts        generation off the main thread
  player.ts        AudioWorklet ring buffer for streaming playback
```

Generation runs in a Web Worker — the per-frame loop would otherwise block the
main thread and stutter playback. Audio is pushed to an `AudioWorklet` ring
buffer rather than scheduled as `AudioBufferSourceNode`s, so chunk boundaries
don't click.

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

WASM works everywhere and is the default. WebGPU is substantially faster where
available, but check numerics before trusting it — WebGPU kernels are not
bit-identical to the CPU path, and this model's sampling happens inside the
graph, so small differences change which token is drawn.
