/**
 * Cross-backend parity: the ggml/GGUF runtime must produce the SAME frame codes
 * as the ONNX runtime for the same text, voice and draw sequence.
 *
 * Both backends take their sampler's uniforms as inputs, so the randomness is
 * the caller's and an exact comparison is possible. At f32 the codes must match
 * element for element — anything less means the port drifted. A *quantized*
 * GGUF is expected to differ (it samples from slightly different logits); this
 * script reports the divergence rather than failing on it, and pins only f32.
 *
 * Needs the WASM build (cpp/build-wasm.sh) and onnxruntime-node:
 *
 *   npm install --no-save onnxruntime-node
 *   node --loader ./test/ts-loader.mjs test/ggml-parity.mjs \
 *        dist/model ../cpp/models/zerotts-f32.gguf
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';

import { MossCodecDecoder } from '../src/codec.ts';
import { ZeroTTSBrowser } from '../src/synthesizer.ts';
import { BpeTokenizer } from '../src/tokenizer.ts';

const here = path.dirname(fileURLToPath(import.meta.url));

const MODEL = process.argv[2] ?? 'dist/model';
const GGUF  = process.argv[3] ?? '../cpp/models/zerotts-f32.gguf';
const TEXT  = process.argv[4] ?? 'Xin chào các bạn, hôm nay trời rất đẹp và tôi muốn đi dạo.';
const VOICE = process.argv[5] ?? 'kimoanh';
const SEED  = Number(process.argv[6] ?? 1234);

const j = (p) => JSON.parse(fs.readFileSync(`${MODEL}/${p}`, 'utf8'));
const buf = (p) => {
  const b = fs.readFileSync(`${MODEL}/${p}`);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
};
function npy(b, kind) {
  const u8 = new Uint8Array(b);
  const dv = new DataView(b);
  const major = u8[6];
  const hlen = major >= 2 ? dv.getUint32(8, true) : dv.getUint16(8, true);
  const off = (major >= 2 ? 12 : 10) + hlen;
  return kind === 'f4' ? new Float32Array(b, off) : new BigInt64Array(b, off);
}

/** mulberry32 — the same generator js/src/rng.ts and cpp/test/bench.cpp use. */
class Rng {
  constructor(seed) { this.s = seed >>> 0; }
  next() {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  fill(out) { for (let i = 0; i < out.length; i++) out[i] = this.next(); return out; }
}

const tokenizer = await BpeTokenizer.create(j('tokenizer.json'));
const voice = new Float32Array(buf(`voices/${VOICE}/voice.bin`));

// ── ONNX ────────────────────────────────────────────────────────────────────
const opts = { executionProviders: ['cpu'] };
const [textEncoder, prefixStep, localFrameDecode] = await Promise.all([
  ort.InferenceSession.create(`${MODEL}/onnx/text_encoder.onnx`, opts),
  ort.InferenceSession.create(`${MODEL}/onnx/prefix_step.onnx`, opts),
  ort.InferenceSession.create(`${MODEL}/onnx/local_frame_decode.onnx`, opts),
]);
const codec = await MossCodecDecoder.create(
  j('onnx/codec/codec_browser_onnx_meta.json'),
  {
    decodeFull: buf('onnx/codec/moss_audio_tokenizer_decode_full.onnx'),
    decodeStep: buf('onnx/codec/moss_audio_tokenizer_decode_step.onnx'),
  },
  {
    ...opts,
    externalData: [{
      path: 'moss_audio_tokenizer_decode_shared.data',
      data: new Uint8Array(buf('onnx/codec/moss_audio_tokenizer_decode_shared.data')),
    }],
  },
);
const onnx = new ZeroTTSBrowser(
  { textEncoder, prefixStep, localFrameDecode }, codec, tokenizer, j('config.json'),
  npy(buf('null_voice_emb.npy'), 'f4'), npy(buf('silence_frame.npy'), 'i8'),
);

const onnxFrames = [];
const tOnnx = performance.now();
for await (const f of onnx.generateFrames(TEXT, voice, {}, SEED)) {
  onnxFrames.push(Array.from(f, Number));
}
const onnxMs = performance.now() - tOnnx;

// ── ggml (the actual WASM artifact the browser loads) ───────────────────────
const { default: createZeroTTS } = await import(
  path.join(here, '..', '..', 'cpp', 'build-wasm', 'zerotts-wasm.js'));
const M = await createZeroTTS({
  locateFile: (f) => path.join(here, '..', '..', 'cpp', 'build-wasm', f),
});

const bytes = fs.readFileSync(GGUF);
const stage = M._zt_alloc(bytes.length);
M.HEAPU8.set(bytes, stage);
const ctx = M._zt_load(stage, bytes.length, 4);
M._zt_dealloc(stage);
if (!ctx) throw new Error(`failed to load ${GGUF}`);

const HP = { D_MODEL: 0, NUM_CODEBOOKS: 8, N_VOICE_QUERIES: 11 };
const K = M._zt_hparam(ctx, HP.NUM_CODEBOOKS);

const ids = Int32Array.from(tokenizer.encode(TEXT), Number);
const idsPtr = M._zt_alloc(ids.length * 4);
const voicePtr = M._zt_alloc(voice.length * 4);
const auPtr = M._zt_alloc(K * 4);
const codePtr = M._zt_alloc(K * 4);
const eoaPtr = M._zt_alloc(4);
M.HEAP32.set(ids, idsPtr >> 2);
M.HEAPF32.set(voice, voicePtr >> 2);
if (M._zt_begin(ctx, idsPtr, ids.length, voicePtr) !== 0) throw new Error('zt_begin failed');

const rng = new Rng(SEED);
const au = new Float32Array(K);
const ggmlFrames = [];
let tailLeft = null;
const tGgml = performance.now();
for (let t = 0; ; t++) {
  const forbidEoa = (t < 4 || tailLeft !== null) ? 1 : 0;
  const cu = rng.next();
  rng.fill(au);
  M.HEAPF32.set(au, auPtr >> 2);
  if (M._zt_frame(ctx, forbidEoa, 1.0, 50, 0.8, 25, 0.95, 1.2, cu, auPtr, codePtr, eoaPtr) !== 0) {
    throw new Error(`zt_frame failed at ${t}`);
  }
  const isEoa = M.HEAP32[eoaPtr >> 2] !== 0;
  const codes = Array.from(M.HEAP32.subarray(codePtr >> 2, (codePtr >> 2) + K));
  if (tailLeft === null && isEoa) tailLeft = 1;
  if ((tailLeft !== null && tailLeft <= 0) || t >= 1500) break;
  ggmlFrames.push(codes);
  if (tailLeft !== null && --tailLeft <= 0) break;
  if (M._zt_advance(ctx, codePtr, t) !== 0) throw new Error(`zt_advance failed at ${t}`);
}
const ggmlMs = performance.now() - tGgml;

// ── compare ─────────────────────────────────────────────────────────────────
const n = Math.min(onnxFrames.length, ggmlFrames.length);
let diff = 0;
for (let i = 0; i < n; i++) {
  for (let k = 0; k < K; k++) if (onnxFrames[i][k] !== ggmlFrames[i][k]) diff++;
}

console.log(`text   ${JSON.stringify(TEXT)}`);
console.log(`voice  ${VOICE}   seed ${SEED}   gguf ${path.basename(GGUF)}`);
console.log(`onnx   ${onnxFrames.length} frames in ${(onnxMs / 1000).toFixed(2)} s`);
console.log(`ggml   ${ggmlFrames.length} frames in ${(ggmlMs / 1000).toFixed(2)} s`);
console.log(`codes  ${diff} of ${n * K} differ over the ${n} shared frames`);

const isF32 = /f32/.test(GGUF);
if (isF32 && (diff !== 0 || onnxFrames.length !== ggmlFrames.length)) {
  console.error('FAIL: an f32 GGUF must reproduce the ONNX frame codes exactly');
  process.exit(1);
}
console.log(isF32 ? 'PASS' : 'quantized build — divergence above is expected, not a failure');
