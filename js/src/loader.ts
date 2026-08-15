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
  /** 'wasm' works everywhere. 'webgpu' is faster but its kernels are not
   *  bit-identical to CPU, and this model samples inside the graph, so small
   *  numeric differences change which token is drawn. */
  executionProvider?: 'wasm' | 'webgpu';
  threads?: number;
}

export interface LoadedModel {
  tts: ZeroTTSBrowser;
  voices: VoiceIndex;
  base: string;
}

export async function loadModel(options: LoadOptions = {}): Promise<LoadedModel> {
  const base = repoBaseUrl(options.repo ?? DEFAULT_REPO, options.revision);
  const ep = options.executionProvider ?? 'wasm';

  ort.env.wasm.numThreads = options.threads ?? Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: [ep],
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

  const tokenizer = await BpeTokenizer.create(tokenizerJson);
  const tts = new ZeroTTSBrowser(
    { textEncoder, prefixStep, localFrameDecode },
    codec, tokenizer, config, parseNpyFloat32(nullVoiceBuf));

  await tts.warmup();
  return { tts, voices, base };
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
function parseNpyFloat32(buffer: ArrayBuffer): Float32Array {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(1, 6));
  if (magic !== 'NUMPY') throw new Error('null_voice_emb.npy: not a .npy file');

  const major = bytes[6];
  const headerLen = major >= 2
    ? new DataView(buffer).getUint32(8, true)
    : new DataView(buffer).getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const header = new TextDecoder().decode(bytes.subarray(headerStart, headerStart + headerLen));

  if (!/'descr':\s*'[<|]f4'/.test(header)) {
    throw new Error(`null_voice_emb.npy: expected little-endian float32, got ${header}`);
  }
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error('null_voice_emb.npy: Fortran order is not supported');
  }
  return new Float32Array(buffer, headerStart + headerLen);
}
