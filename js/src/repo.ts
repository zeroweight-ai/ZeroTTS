/**
 * Where the model files live, and the small fetches that do not need a runtime.
 *
 * Split out of loader.ts on purpose: loader.ts pulls in onnxruntime-web and the
 * synthesizer, and everything in this file is needed by the PAGE (repo URLs, the
 * size note, voice previews) as well as by the worker. Importing it from main.ts
 * must not drag the runtime onto the UI thread.
 */

import { fetchWithCache, isCached, totalBytes } from './cache';

/**
 * Which runtime generates frames.
 *
 * `ggml` is the default: a WASM build of ggml reading a quantized GGUF (see
 * ../../cpp/), which downloads ~4x less than the ONNX graphs and is faster from
 * four threads up. `onnx` is onnxruntime-web, the original path — kept because
 * it is the one with a Python reference implementation behind it, and because
 * it is the only one that implements CFG.
 */
export type Backend = 'ggml' | 'onnx';

export const DEFAULT_BACKEND: Backend = 'ggml';

/** GGUF weights for the ggml backend. */
export const GGUF_REPO = 'zeroweight-ai/ZeroTTS-GGUF';
/** ONNX graphs for the onnxruntime backend. */
export const ONNX_REPO = 'zeroweight-ai/ZeroTTS';

/**
 * Which GGUF to fetch. f32 by default, for two reasons that both cut against
 * the usual instinct to ship the quantized one:
 *
 *  - **It is the fastest build**, at every thread count measured. WASM's SIMD
 *    has no integer dot-product instruction, so a quantized block costs more
 *    arithmetic than it saves in bandwidth; quantization here is a
 *    download-size lever, not a speed one.
 *  - **It is bit-exact.** f32 reproduces the ONNX runtime's frame codes
 *    exactly. q8_0 draws a different code 4.9% of the time and q4_0 36.5% —
 *    not broken, but a different take of the same sentence.
 *
 * At 772 MB it is still a smaller download than the ~858 MB of ONNX graphs it
 * replaces. Pick q8_0 (206 MB) or q4_0 (124 MB) when download size is the
 * binding constraint and the drift is acceptable. See ../../cpp/README.md.
 */
export const DEFAULT_GGUF = 'gguf/zerotts-f32.gguf';

/** The GGUF builds the model repository ships, smallest first. `drift` is the
 *  share of sampled codes that come out differently from the unquantized model
 *  when both are driven along the same code sequence. */
export const GGUF_BUILDS = [
  { file: 'gguf/zerotts-q4_0.gguf', label: 'q4_0 — 124 MB, nhẹ nhất, sai khác 36%' },
  { file: 'gguf/zerotts-q8_0.gguf', label: 'q8_0 — 206 MB, nhẹ, sai khác 4,9%' },
  { file: 'gguf/zerotts-f32.gguf', label: 'f32 — 772 MB, nhanh nhất, chính xác tuyệt đối (mặc định)' },
] as const;

export function defaultRepo(backend: Backend): string {
  return backend === 'ggml' ? GGUF_REPO : ONNX_REPO;
}

/** @deprecated prefer defaultRepo(backend); kept so callers that predate the
 *  backend switch still resolve to something sensible. */
export const DEFAULT_REPO = GGUF_REPO;

export function repoBaseUrl(repo: string, revision = 'main'): string {
  if (repo.startsWith('http://') || repo.startsWith('https://') || repo.startsWith('/')) {
    return repo.replace(/\/$/, '');
  }
  return `https://huggingface.co/${repo}/resolve/${revision}`;
}

/** The codec decoder is ONNX on both backends: ~45 MB, and not on the
 *  per-frame hot path, so there was nothing to gain from porting it. */
const CODEC = [
  'onnx/codec/moss_audio_tokenizer_decode_full.onnx',
  'onnx/codec/moss_audio_tokenizer_decode_step.onnx',
  'onnx/codec/moss_audio_tokenizer_decode_shared.data',
];

const ONNX_GRAPHS = [
  'onnx/prefix_step.onnx',
  'onnx/local_frame_decode.onnx',
  'onnx/text_encoder.onnx',
];

/** The big files, in fetch order — what the progress bar and size note count. */
export function modelFiles(backend: Backend, gguf = DEFAULT_GGUF): string[] {
  return backend === 'ggml' ? [gguf, ...CODEC] : [...ONNX_GRAPHS, ...CODEC];
}

export function modelUrls(base: string, backend: Backend = DEFAULT_BACKEND,
                          gguf = DEFAULT_GGUF): string[] {
  return modelFiles(backend, gguf).map((g) => `${base}/${g}`);
}

/** A voice's preview clip, for the picker. */
export function voicePreviewUrl(base: string, name: string): string {
  return `${base}/voices/${name}/preview.wav`;
}

/** Total download size, and whether it is already cached — for the size warning. */
export async function downloadInfo(
  base: string, backend: Backend = DEFAULT_BACKEND, gguf = DEFAULT_GGUF,
): Promise<{ bytes: number; cached: boolean }> {
  const urls = modelUrls(base, backend, gguf);
  const [bytes, cached] = await Promise.all([totalBytes(urls), isCached(urls)]);
  return { bytes, cached };
}

/** Load a voice's latents. `voice.bin` is raw little-endian float32 — the repo
 *  ships it precisely so the browser needs no zip/npy parser. */
export async function loadVoice(base: string, name: string): Promise<Float32Array> {
  const buf = await fetchWithCache(`${base}/voices/${name}/voice.bin`);
  return new Float32Array(buf);
}
