/**
 * Where the model files live, and the small fetches that do not need a runtime.
 *
 * Split out of loader.ts on purpose: loader.ts pulls in onnxruntime-web and the
 * synthesizer, and everything in this file is needed by the PAGE (repo URLs, the
 * size note, voice previews) as well as by the worker. Importing it from main.ts
 * must not drag the runtime onto the UI thread.
 */

import { fetchWithCache, isCached, totalBytes } from './cache';

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

/** A voice's preview clip, for the picker. */
export function voicePreviewUrl(base: string, name: string): string {
  return `${base}/voices/${name}/preview.wav`;
}

/** Total download size, and whether it is already cached — for the size warning. */
export async function downloadInfo(base: string): Promise<{ bytes: number; cached: boolean }> {
  const urls = modelUrls(base);
  const [bytes, cached] = await Promise.all([totalBytes(urls), isCached(urls)]);
  return { bytes, cached };
}

/** Load a voice's latents. `voice.bin` is raw little-endian float32 — the repo
 *  ships it precisely so the browser needs no zip/npy parser. */
export async function loadVoice(base: string, name: string): Promise<Float32Array> {
  const buf = await fetchWithCache(`${base}/voices/${name}/voice.bin`);
  return new Float32Array(buf);
}
