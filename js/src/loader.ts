/**
 * Resolving and loading a ZeroTTS model in the browser.
 *
 * Files come straight from the Hugging Face CDN (or any static host with the
 * same layout). Nothing is uploaded anywhere — the model runs on the viewer's
 * machine.
 */

import * as ort from 'onnxruntime-web';

import { fetchWithCache, isCached, ProgressFn, totalBytes } from './cache';
import { CodecMeta, MossCodecDecoder } from './codec';
import { BpeTokenizer } from './tokenizer';
import { ZeroTTSBrowser } from './synthesizer';
import { VoiceIndex, ZeroTTSConfig } from './types';

export const DEFAULT_REPO = 'zeroweight-ai/ZeroTTS';

export function repoBaseUrl(repo: string, revision = 'main'): string {
  if (repo.startsWith('http://') || repo.startsWith('https://') || repo.startsWith('/')) {
    return repo.replace(/\/$/, '');
  }
  return `https://huggingface.co/${repo}/resolve/${revision}`;
}

const GRAPHS = [
  'onnx/prefix_step.onnx',
  'onnx/local_frame_decode.onnx',
  'onnx/text_encoder.onnx',
  'onnx/codec/moss_audio_tokenizer_decode_full.onnx',
  'onnx/codec/moss_audio_tokenizer_decode_step.onnx',
  'onnx/codec/moss_audio_tokenizer_decode_shared.data',
];

/** A voice's preview clip, for the picker. */
export function voicePreviewUrl(base: string, name: string): string {
  return `${base}/voices/${name}/preview.wav`;
}

export function modelUrls(base: string): string[] {
  return GRAPHS.map((g) => `${base}/${g}`);
}

/** Total download size, and whether it is already cached — for the size warning. */
export async function downloadInfo(base: string): Promise<{ bytes: number; cached: boolean }> {
  const urls = modelUrls(base);
  const [bytes, cached] = await Promise.all([totalBytes(urls), isCached(urls)]);
  return { bytes, cached };
}

export interface LoadOptions {
  repo?: string;
  revision?: string;
  onProgress?: ProgressFn;
  threads?: number;
}

export interface LoadedModel {
  tts: ZeroTTSBrowser;
  voices: VoiceIndex;
  base: string;
}

export async function loadModel(options: LoadOptions = {}): Promise<LoadedModel> {
  const base = repoBaseUrl(options.repo ?? DEFAULT_REPO, options.revision);

  ort.env.wasm.numThreads = options.threads ?? Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;

  // WASM only. WebGPU's kernels are not bit-identical to the CPU path, and this
  // model samples INSIDE the graph — small numeric differences change which
  // token is drawn, so the provider is not a free speed knob but a change in
  // output quality. It is deliberately not offered.
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  };

  const overall = { loaded: 0, total: await totalBytes(modelUrls(base)) };
  const get = (path: string) =>
    fetchWithCache(`${base}/${path}`, options.onProgress, overall);
  const getJson = async (path: string) => {
    const buf = await fetch(`${base}/${path}`).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    return JSON.parse(new TextDecoder().decode(buf));
  };

  const [config, tokenizerJson, codecMeta, voices] = await Promise.all([
    getJson('config.json') as Promise<ZeroTTSConfig>,
    getJson('tokenizer.json'),
    getJson('onnx/codec/codec_browser_onnx_meta.json') as Promise<CodecMeta>,
    getJson('voices/index.json').catch(() => ({ voices: [] })) as Promise<VoiceIndex>,
  ]);

  // The codec decoder graphs share one external .data file; ORT must be told
  // about it explicitly — it is not fetched implicitly.
  const sharedData = await get('onnx/codec/moss_audio_tokenizer_decode_shared.data');
  const codecOptions: ort.InferenceSession.SessionOptions = {
    ...sessionOptions,
    externalData: [{
      path: 'moss_audio_tokenizer_decode_shared.data',
      data: sharedData,
    }],
  };

  const [prefixBuf, localBuf, textBuf, decodeFull, decodeStep, nullVoiceBuf] =
    await Promise.all([
      get('onnx/prefix_step.onnx'),
      get('onnx/local_frame_decode.onnx'),
      get('onnx/text_encoder.onnx'),
      get('onnx/codec/moss_audio_tokenizer_decode_full.onnx'),
      get('onnx/codec/moss_audio_tokenizer_decode_step.onnx'),
      get('null_voice_emb.npy'),
    ]);

  const [prefixStep, localFrameDecode, textEncoder, codec] = await Promise.all([
    ort.InferenceSession.create(prefixBuf, sessionOptions),
    ort.InferenceSession.create(localBuf, sessionOptions),
    ort.InferenceSession.create(textBuf, sessionOptions),
    MossCodecDecoder.create(codecMeta, { decodeFull, decodeStep }, codecOptions),
  ]);

  // The codec's canonical silence frame, used to pad between segments. Absent
  // in model directories built before it was shipped; the synthesizer then
  // skips padding rather than guessing (zeros are a real code, not silence).
  let silenceFrame: BigInt64Array | null = null;
  try {
    silenceFrame = parseNpyInt64(await get('silence_frame.npy'));
  } catch {
    console.warn('silence_frame.npy missing — segments will not be padded');
  }

  const tokenizer = await BpeTokenizer.create(tokenizerJson);
  const tts = new ZeroTTSBrowser(
    { textEncoder, prefixStep, localFrameDecode },
    codec, tokenizer, config, parseNpyFloat32(nullVoiceBuf), silenceFrame);

  await tts.warmup();
  return { tts, voices, base };
}

/**
 * Sample texts for the demo's picker.
 *
 * Imported as a raw string from the same webui/test_samples.txt the Python UI
 * reads, so the two demos stay in step and the file has one home. Vite inlines
 * it at build time (see vite.config.ts's `server.fs.allow`, which lets the
 * import reach outside js/).
 */
export async function loadSampleTexts(): Promise<Record<string, string>> {
  try {
    const [{ parseSamples }, raw] = await Promise.all([
      import('./chunking'),
      import('../../webui/test_samples.txt?raw'),
    ]);
    return parseSamples((raw as { default: string }).default);
  } catch {
    return {};
  }
}

/** Load a voice's latents. `voice.bin` is raw little-endian float32 — the repo
 *  ships it precisely so the browser needs no zip/npy parser. */
export async function loadVoice(base: string, name: string): Promise<Float32Array> {
  const buf = await fetchWithCache(`${base}/voices/${name}/voice.bin`);
  return new Float32Array(buf);
}

/**
 * Minimal .npy reader for the one array we load that way (null_voice_emb.npy).
 * Only handles the little-endian float32, C-order case that file is written in;
 * anything else is a repo problem worth failing loudly on.
 */
function npyPayload(buffer: ArrayBuffer, descr: RegExp, what: string): [number, string] {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error(`${what}: not a .npy file`);
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLen = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const header = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen));

  if (!descr.test(header)) throw new Error(`${what}: unexpected dtype — ${header}`);
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error(`${what}: Fortran order is not supported`);
  }
  return [headerStart + headerLen, header];
}

function parseNpyFloat32(buffer: ArrayBuffer): Float32Array {
  const [offset] = npyPayload(buffer, /'descr':\s*'[<|]f4'/, 'null_voice_emb.npy');
  return new Float32Array(buffer, offset);
}

function parseNpyInt64(buffer: ArrayBuffer): BigInt64Array {
  const [offset] = npyPayload(buffer, /'descr':\s*'[<|]i8'/, 'silence_frame.npy');
  // A BigInt64Array view requires an 8-byte-aligned offset. NumPy pads its
  // header to a 64-byte boundary so this normally holds, but a hand-written or
  // re-saved file need not — copy rather than throw a RangeError nobody can act
  // on.
  if (offset % 8 === 0) return new BigInt64Array(buffer, offset);
  return new BigInt64Array(buffer.slice(offset));
}
