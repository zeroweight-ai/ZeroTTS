/**
 * End-to-end check: the browser synthesizer must produce the SAME frame codes
 * as the Python package.
 *
 * Sampling happens inside `local_frame_decode.onnx` and takes its random draws
 * as graph *inputs*, so the caller owns the randomness. That makes an exact
 * cross-language comparison possible: record the draws this port consumes,
 * replay them in Python, and the frame codes must match element for element.
 * Nothing weaker is meaningful — the audio is sampled, so two runs of the same
 * text differ unless the draws are pinned.
 *
 * Needs a local model directory and onnxruntime-node (the browser's ORT build
 * cannot run outside a browser):
 *
 *   npm install --no-save onnxruntime-node
 *   node --experimental-strip-types test/frames.mjs /path/to/model
 *
 * Writes js_frames.json + js_draws.json, then:
 *   python test/py_frames.py /path/to/model   # writes py_frames.json, compares
 */

import fs from 'node:fs';
import * as ort from 'onnxruntime-node';

import { MossCodecDecoder } from '../src/codec.ts';
import { Rng } from '../src/rng.ts';
import { ZeroTTSBrowser } from '../src/synthesizer.ts';
import { BpeTokenizer } from '../src/tokenizer.ts';

const M = process.argv[2];
if (!M) throw new Error('usage: node test/frames.mjs <model-dir>');
const TEXT = process.argv[3] ?? 'Xin chào các bạn.';
const VOICE = process.argv[4] ?? 'arya';
const SEED = 1234;

const j = (p) => JSON.parse(fs.readFileSync(`${M}/${p}`, 'utf8'));
const buf = (p) => {
  const b = fs.readFileSync(`${M}/${p}`);
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

const opts = { executionProviders: ['cpu'] };
const [textEncoder, prefixStep, localFrameDecode] = await Promise.all([
  ort.InferenceSession.create(`${M}/onnx/text_encoder.onnx`, opts),
  ort.InferenceSession.create(`${M}/onnx/prefix_step.onnx`, opts),
  ort.InferenceSession.create(`${M}/onnx/local_frame_decode.onnx`, opts),
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
      // onnxruntime-node wants a typed array here; ORT-web accepts an
      // ArrayBuffer. Only this harness cares.
      data: new Uint8Array(buf('onnx/codec/moss_audio_tokenizer_decode_shared.data')),
    }],
  },
);

const tts = new ZeroTTSBrowser(
  { textEncoder, prefixStep, localFrameDecode }, codec,
  await BpeTokenizer.create(j('tokenizer.json')), j('config.json'),
  npy(buf('null_voice_emb.npy'), 'f4'), npy(buf('silence_frame.npy'), 'i8'),
);

// Record every draw so Python can replay the identical sequence.
const draws = [];
const next = Rng.prototype.next;
Rng.prototype.next = function patched() { const v = next.call(this); draws.push(v); return v; };

const voice = new Float32Array(buf(`voices/${VOICE}/voice.bin`));
const frames = [];
for await (const f of tts.generateFrames(TEXT, voice, {}, SEED)) {
  frames.push(Array.from(f, Number));
}

fs.writeFileSync('js_frames.json', JSON.stringify(frames));
fs.writeFileSync('js_draws.json', JSON.stringify(draws));
console.log(`${frames.length} frames, ${draws.length} draws -> js_frames.json, js_draws.json`);
console.log(`now run: python test/py_frames.py ${M} ${JSON.stringify(TEXT)} ${VOICE}`);
