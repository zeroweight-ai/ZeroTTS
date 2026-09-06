/**
 * Backend A/B for bench-ggml.html.
 *
 * Runs frame generation only — no codec, no audio — because that is where all
 * the time goes and it is the only part the ggml port replaces. Both backends
 * get the same text, voice, seed and sampling options, so the frame codes can
 * be compared alongside the timings: at f32 they must match exactly, and a
 * quantized GGUF is expected to diverge (it samples from slightly different
 * logits), which the report states rather than hides.
 */

import * as ort from 'onnxruntime-web';

import { CodecMeta, MossCodecDecoder } from './codec';
import { ZeroTTSGgml } from './ggmlBackend';
import { ZeroTTSBrowser } from './synthesizer';
import { BpeTokenizer } from './tokenizer';
import { ZeroTTSConfig } from './types';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const log = (s: string) => { $('log').textContent += s + '\n'; };

$('coi').textContent = String(typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
$('hc').textContent = String(navigator.hardwareConcurrency);

// ?model=<path> points at a different export, for comparing graph revisions.
const MODEL = new URLSearchParams(location.search).get('model') ?? '/dist/model';
const SEED = 1234;

const json = <T>(p: string) => fetch(`${MODEL}/${p}`).then((r) => r.json() as Promise<T>);
const bin = (p: string) => fetch(`${MODEL}/${p}`).then((r) => r.arrayBuffer());

/** Little-endian .npy reader — only the two small side files need it. */
function npy(buf: ArrayBuffer, kind: 'f4' | 'i8') {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  const major = u8[6];
  const headerLen = major >= 2 ? dv.getUint32(8, true) : dv.getUint16(8, true);
  const off = (major >= 2 ? 12 : 10) + headerLen;
  return kind === 'f4' ? new Float32Array(buf, off) : new BigInt64Array(buf, off);
}

interface Result {
  backend: string;
  frames: number[][];
  wallMs: number;
  detail: string;
}

const results: Result[] = [];

function report() {
  const audioS = (r: Result) => r.frames.length / 12.5;
  const rows = results.map((r) => `
    <tr><td>${r.backend}</td><td>${r.frames.length}</td>
        <td>${audioS(r).toFixed(2)} s</td>
        <td>${(r.wallMs / 1000).toFixed(2)} s</td>
        <td>${(r.wallMs / 1000 / audioS(r)).toFixed(3)}</td>
        <td><b>${(audioS(r) / (r.wallMs / 1000)).toFixed(2)}x</b></td>
        <td>${r.detail}</td></tr>`).join('');
  $('results').innerHTML = `<table>
    <tr><th>backend</th><th>frames</th><th>audio</th><th>wall</th><th>RTF</th>
        <th>realtime</th><th>per-stage</th></tr>${rows}</table>`;

  // Same seed and same draw order, so identical codes are the expected result
  // whenever both sides are running f32.
  if (results.length >= 2) {
    const [a, b] = [results[results.length - 2], results[results.length - 1]];
    const n = Math.min(a.frames.length, b.frames.length);
    let diff = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < a.frames[i].length; k++) if (a.frames[i][k] !== b.frames[i][k]) diff++;
    }
    log(`${a.backend} vs ${b.backend}: ${diff} differing codes over ${n} shared frames`
      + (a.frames.length === b.frames.length ? '' : ` (frame counts differ: ${a.frames.length} vs ${b.frames.length})`));
  }
}

// Only ONE backend is held at a time. Each is ~0.2-0.9 GB of buffers plus a
// WASM heap, and keeping two (or two GGUFs) alive pushes the renderer into
// swapping — which does not fail, it just makes every subsequent measurement
// 10-20x slower and looks exactly like a real regression. Releasing first is
// what keeps successive runs on this page comparable.
let ortTts: ZeroTTSBrowser | null = null;
let ortSessions: ort.InferenceSession[] = [];
let ggml: { key: string; tts: ZeroTTSGgml } | null = null;

async function releaseAll(): Promise<void> {
  for (const s of ortSessions) {
    await s.release().catch(() => {});
  }
  ortSessions = [];
  ortTts = null;
  ggml?.tts.free();
  ggml = null;
}

async function loadOnnx(threads: number): Promise<ZeroTTSBrowser> {
  if (ortTts) return ortTts;
  await releaseAll();
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.simd = true;
  const opts: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  };
  log('loading ONNX graphs …');
  const [textEncoder, prefixStep, localFrameDecode] = await Promise.all([
    bin('onnx/text_encoder.onnx').then((b) => ort.InferenceSession.create(b, opts)),
    bin('onnx/prefix_step.onnx').then((b) => ort.InferenceSession.create(b, opts)),
    bin('onnx/local_frame_decode.onnx').then((b) => ort.InferenceSession.create(b, opts)),
  ]);
  const codec = await MossCodecDecoder.create(
    await json<CodecMeta>('onnx/codec/codec_browser_onnx_meta.json'),
    {
      decodeFull: await bin('onnx/codec/moss_audio_tokenizer_decode_full.onnx'),
      decodeStep: await bin('onnx/codec/moss_audio_tokenizer_decode_step.onnx'),
    },
    {
      ...opts,
      externalData: [{
        path: 'moss_audio_tokenizer_decode_shared.data',
        data: await bin('onnx/codec/moss_audio_tokenizer_decode_shared.data'),
      }],
    },
  );
  ortSessions = [textEncoder, prefixStep, localFrameDecode];
  ortTts = new ZeroTTSBrowser(
    { textEncoder, prefixStep, localFrameDecode }, codec,
    await BpeTokenizer.create(await json('tokenizer.json')),
    await json<ZeroTTSConfig>('config.json'),
    npy(await bin('null_voice_emb.npy'), 'f4') as Float32Array,
    npy(await bin('silence_frame.npy'), 'i8') as BigInt64Array,
  );
  return ortTts;
}

async function loadGgml(file: string, threads: number): Promise<ZeroTTSGgml> {
  const key = `${file}:${threads}`;
  if (ggml?.key === key) return ggml.tts;
  await releaseAll();
  log(`loading ${file} …`);
  const t0 = performance.now();
  const gguf = await fetch(`/ggml/models/${file}`).then((r) => r.arrayBuffer());
  const tts = await ZeroTTSGgml.create(
    gguf, await BpeTokenizer.create(await json('tokenizer.json')), threads);
  log(`  ${(gguf.byteLength / 1e6).toFixed(0)} MB, ready in ${(performance.now() - t0).toFixed(0)} ms`);
  ggml = { key, tts };
  return tts;
}

async function runOnnx(threads: number, text: string, voiceName: string) {
  {
    const tts = await loadOnnx(threads);
    const voice = new Float32Array(await bin(`voices/${voiceName}/voice.bin`));
    log(`running onnxruntime-web (${threads} threads) …`);
    const frames: number[][] = [];
    const t0 = performance.now();
    for await (const f of tts.generateFrames(text, voice, {}, SEED)) {
      frames.push(Array.from(f, Number));
    }
    results.push({
      backend: `onnxruntime-web (${threads}t)`, frames,
      wallMs: performance.now() - t0, detail: '—',
    });
    report();
  }
}

async function runGgml(file: string, threads: number, text: string, voiceName: string) {
  {
    const tts = await loadGgml(file, threads);
    const voice = new Float32Array(await bin(`voices/${voiceName}/voice.bin`));
    log(`running ggml ${file} (${threads} threads) …`);
    tts.resetTimings();
    const frames: number[][] = [];
    const t0 = performance.now();
    for await (const f of tts.generateFrames(text, voice, {}, SEED)) {
      frames.push(Array.from(f));
    }
    const wallMs = performance.now() - t0;
    const tm = tts.timings();
    results.push({
      backend: `ggml ${file.replace(/^zerotts-|\.gguf$/g, '')} (${threads}t)`, frames, wallMs,
      detail: `begin ${tm.beginMs.toFixed(0)} ms · frame ${(tm.frameMs / tm.frames).toFixed(2)} ms · `
            + `advance ${(tm.advanceMs / tm.frames).toFixed(2)} ms`,
    });
    report();
  }
}

const inputs = () => ({
  threads: Number(($('threads') as HTMLInputElement).value),
  text: ($('text') as HTMLInputElement).value,
  voice: ($('voice') as HTMLInputElement).value,
  gguf: ($('gguf') as HTMLSelectElement).value,
});

$('run-onnx').addEventListener('click', async () => {
  const i = inputs();
  try { await runOnnx(i.threads, i.text, i.voice); }
  catch (e) { log(`onnx error: ${(e as Error).message}`); }
});

$('run-ggml').addEventListener('click', async () => {
  const i = inputs();
  try { await runGgml(i.gguf, i.threads, i.text, i.voice); }
  catch (e) { log(`ggml error: ${(e as Error).message}`); }
});

/**
 * Scripted entry point, so a sweep can be driven from the console or a test
 * runner instead of by clicking. onnxruntime-web's thread count is a global set
 * before the first session is created, so a sweep over thread counts has to
 * reload the page between ONNX runs — hence one thread count per page load.
 */
declare global { interface Window { zbench: unknown } }
window.zbench = {
  onnx: runOnnx,
  ggml: runGgml,
  results,
  table: () => results.map((r) => ({
    backend: r.backend,
    frames: r.frames.length,
    realtime: +(r.frames.length / 12.5 / (r.wallMs / 1000)).toFixed(2),
    detail: r.detail,
  })),
};
