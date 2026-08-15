# ZeroTTS browser demo

Runs ZeroTTS client-side with [`onnxruntime-web`](https://onnxruntime.ai/). No
server, no upload — weights are fetched from the Hugging Face CDN and everything
executes on the viewer's machine.

This is a **demo in this repository**, not a published npm package.

```bash
npm install
npm run dev        # http://localhost:5173
npm run typecheck
npm run build      # static bundle in dist/
```

> **Not yet run end-to-end.** The Python runtime is verified bit-identical
> against the reference implementation; this port is a faithful translation of
> the same contract but has not been executed against real weights. Treat it as
> a starting point, and run the parity check below before trusting it.

## The download

The weights are fp32 and unquantized — about **900 MB** total, fetched once and
persisted via the Cache API. That is a deliberate quality-over-size decision
(see [../docs/BROWSER.md](../docs/BROWSER.md)); the UI states the size before
downloading. Not suitable for mobile data.

## Layout

| File | Role |
|---|---|
| `src/synthesizer.ts` | the two-calls-per-frame loop — see [../docs/RUNTIME.md](../docs/RUNTIME.md) |
| `src/codec.ts` | MOSS decoder: batch + KV-cached streaming |
| `src/tokenizer.ts` | BPE over `tokenizer.json` |
| `src/loader.ts` | resolve repo URLs, create sessions, load voices |
| `src/cache.ts` | download progress + Cache API persistence |
| `src/player.ts` | AudioWorklet ring buffer, WAV export |
| `src/rng.ts` | seedable PRNG — the sampler's draws are graph *inputs* |
| `src/main.ts` | demo UI wiring |

## Parity check (do this first)

Sampling happens inside `local_frame_decode.onnx` and takes its random draws as
graph inputs, so the caller owns the randomness. That makes an exact
cross-language comparison possible, and it is the only practical way to keep
this port correct across re-exports:

1. In Python, seed `numpy.random` and record the exact draw sequence and the
   resulting frame codes for a fixed text + voice.
2. Feed the same draws through `Rng` (or stub it with the recorded values) and
   assert the JS frame codes match element-for-element.

Frame codes must match exactly. Audio samples may differ in the last bits at
chunk seams when comparing streaming to batch decode — that is the KV-cached
decoder, not an error — but the codes must not.

## Cross-origin isolation

`vite.config.ts` sets COOP/COEP headers so `SharedArrayBuffer` is available and
onnxruntime-web can use multi-threaded WASM. Without them it silently falls back
to a single thread and generation is several times slower. **Whatever hosts the
built bundle must send the same two headers** — GitHub Pages does not, so a Pages
deployment will be slow unless you add a service-worker shim.

## Known gaps

- Generation runs on the main thread. It should move into a Web Worker
  (`worker.ts`) — the per-frame loop otherwise competes with rendering and can
  stutter playback under load.
- WebGPU is selectable but unvalidated. Its kernels are not bit-identical to the
  CPU path, and since this model samples *inside* the graph, small numeric
  differences change which token is drawn. Compare against WASM before using it.
- No long-form chunking. The Python package has `zerotts.chunking`; the browser
  demo synthesizes whatever it is given as one utterance.
