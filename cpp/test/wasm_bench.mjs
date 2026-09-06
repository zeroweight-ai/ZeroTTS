// Runs the WASM build end to end and prints frame codes + timings, so the
// browser build can be checked against the native one (identical codes) and
// against the ONNX path's numbers (js/test/frames.mjs).
//
//   node --experimental-wasm-threads test/wasm_bench.mjs model.gguf voice.bin ids.txt [seed] [threads]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import createZeroTTS from '../build-wasm/zerotts-wasm.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const [gguf, voicePath, idsPath] = process.argv.slice(2);
const seed    = Number(process.argv[5] ?? 1234);
const threads = Number(process.argv[6] ?? 4);
if (!gguf) { console.error('usage: wasm_bench.mjs model.gguf voice.bin ids.txt [seed] [threads]'); process.exit(1); }

// mulberry32 — identical to js/src/rng.ts and cpp/test/bench.cpp.
class Rng {
  constructor(seed) { this.s = seed >>> 0; }
  next() {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

const M = await createZeroTTS({
  locateFile: (f) => path.join(here, '..', 'build-wasm', f),
});

const bytes = fs.readFileSync(gguf);
const tLoad0 = performance.now();
const modelPtr = M._zt_alloc(bytes.length);
M.HEAPU8.set(bytes, modelPtr);
const ctx = M._zt_load(modelPtr, bytes.length, threads);
M._zt_dealloc(modelPtr);
if (!ctx) { console.error('zt_load failed'); process.exit(1); }
const loadMs = performance.now() - tLoad0;

const HP = { N_DEPTH: 7, NUM_CODEBOOKS: 8, N_VOICE_QUERIES: 11, D_MODEL: 0 };
const K = M._zt_hparam(ctx, HP.NUM_CODEBOOKS);
const V = M._zt_hparam(ctx, HP.N_VOICE_QUERIES);
const D = M._zt_hparam(ctx, HP.D_MODEL);

const ids = fs.readFileSync(idsPath, 'utf8').trim().split(/\s+/).map(Number);
const voice = new Float32Array(fs.readFileSync(voicePath).buffer.slice(0));
if (voice.length !== V * D) { console.error(`voice.bin has ${voice.length} floats, expected ${V * D}`); process.exit(1); }

const idsPtr   = M._zt_alloc(ids.length * 4);
const voicePtr = M._zt_alloc(voice.length * 4);
const auPtr    = M._zt_alloc(K * 4);
const codePtr  = M._zt_alloc(K * 4);
const eoaPtr   = M._zt_alloc(4);
const timePtr  = M._zt_alloc(4 * 8);
M.HEAP32.set(ids, idsPtr >> 2);
M.HEAPF32.set(voice, voicePtr >> 2);

const t0 = performance.now();
if (M._zt_begin(ctx, idsPtr, ids.length, voicePtr) !== 0) { console.error('zt_begin failed'); process.exit(1); }

const rng = new Rng(seed);
const frames = [];
let tailLeft = null;
const MIN_FRAMES = 4, MAX_FRAMES = 1500, EOA_EXTRA = 1;

for (let t = 0; ; t++) {
  const forbidEoa = (t < MIN_FRAMES || tailLeft !== null) ? 1 : 0;
  const cu = rng.next();
  const au = new Float32Array(K);
  for (let i = 0; i < K; i++) au[i] = rng.next();
  M.HEAPF32.set(au, auPtr >> 2);

  if (M._zt_frame(ctx, forbidEoa, 1.0, 50, 0.8, 25, 0.95, 1.2, cu, auPtr, codePtr, eoaPtr) !== 0) {
    console.error(`zt_frame failed at t=${t}`); process.exit(1);
  }
  const isEoa = M.HEAP32[eoaPtr >> 2] !== 0;
  const codes = Array.from(M.HEAP32.subarray(codePtr >> 2, (codePtr >> 2) + K));

  if (tailLeft === null && isEoa) tailLeft = EOA_EXTRA;
  if ((tailLeft !== null && tailLeft <= 0) || t >= MAX_FRAMES) break;
  frames.push(codes);
  if (tailLeft !== null && --tailLeft <= 0) break;
  if (M._zt_advance(ctx, codePtr, t) !== 0) { console.error(`zt_advance failed at t=${t}`); process.exit(1); }
}
const wall = (performance.now() - t0) / 1000;

M._zt_timings(ctx, timePtr);
const [textMs, frameMs, advMs, nf] = Array.from(new Float64Array(M.HEAPU8.buffer, timePtr, 4));
const audioS = frames.length / 12.5;

console.error(`load           ${loadMs.toFixed(0)} ms`);
console.error(`${frames.length} frames = ${audioS.toFixed(2)} s audio in ${wall.toFixed(2)} s wall`);
console.error(`RTF = ${(wall / audioS).toFixed(3)}  (${(audioS / wall).toFixed(2)}x realtime)`);
console.error(`  begin (text+prefix) ${textMs.toFixed(1)} ms`);
console.error(`  frame  (local x17)  ${frameMs.toFixed(1)} ms   ${(frameMs / nf).toFixed(2)} ms/frame`);
console.error(`  advance (decoder)   ${advMs.toFixed(1)} ms   ${(advMs / nf).toFixed(2)} ms/frame`);

process.stdout.write(JSON.stringify(frames));
M._zt_free(ctx);
