/**
 * The ggml/GGUF backend — a second implementation of the generation loop that
 * runs the model from a quantized GGUF through a WASM build of ggml instead of
 * through onnxruntime-web.
 *
 * It is a drop-in alternative to `ZeroTTSBrowser` for frame generation only:
 * same `generateFrames` signature, same draw sequence, same frame codes. The
 * codec (waveform decoding) is not part of this and still runs on ORT — it is
 * ~45 MB and not on the per-frame hot path.
 *
 * **Verified bit-exact against the ONNX path**, native and in WASM, for the same
 * text/voice/seed at f32 (see cpp/test/). Quantized builds sample from slightly
 * different logits and so diverge in *which* token is drawn — that is expected
 * and is not a correctness failure, but it does mean the parity fixture only
 * pins the f32 build.
 *
 * Two things this does that the ONNX graphs cannot:
 *
 *  - the per-layer cross-attention K/V over the text are projected **once per
 *    segment**. `prefix_step.onnx` takes `text_states` as an input on every
 *    frame and so redoes 9 x 2 x L x 768 x 768 of projection per frame.
 *  - the control head reads two rows of the 8192-row embedding table rather
 *    than multiplying by all of it, since only <slot> and <eoa> are legal.
 *
 * CFG (`cfgScale > 1`) is **not** implemented here — it needs a batch-2 decoder
 * and the demo defaults it off. `generateFrames` throws rather than silently
 * ignoring it.
 */

import { Rng } from './rng';
import { DEFAULT_SAMPLING, SamplingOptions } from './types';
import { BpeTokenizer } from './tokenizer';

/** Selector values for `zt_hparam`; must match the enum in cpp/src/wasm.cpp. */
const HP = {
  D_MODEL: 0, N_HEADS: 1, N_LAYERS: 2, D_FF: 3,
  LOCAL_N_LAYERS: 4, LOCAL_N_HEADS: 5, LOCAL_D_FF: 6, N_DEPTH: 7,
  NUM_CODEBOOKS: 8, CODEBOOK_SIZE: 9, VOCAB_SIZE: 10,
  N_VOICE_QUERIES: 11, MAX_FRAMES: 12,
} as const;

interface ZeroTTSWasm {
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPF32: Float32Array;
  _zt_alloc(n: number): number;
  _zt_dealloc(p: number): void;
  _zt_load(data: number, size: number, threads: number): number;
  _zt_free(ctx: number): void;
  _zt_hparam(ctx: number, which: number): number;
  _zt_begin(ctx: number, ids: number, n: number, voice: number): number;
  _zt_frame(ctx: number, forbidEoa: number, textTemp: number, textTopK: number,
            audioTemp: number, audioTopK: number, audioTopP: number,
            audioRepPenalty: number, ctrlU: number, audioU: number,
            outCodes: number, outIsEoa: number): number;
  _zt_advance(ctx: number, codes: number, t: number): number;
  _zt_timings(ctx: number, out: number): void;
  _zt_reset_timings(ctx: number): void;
}

/** Where the Emscripten pair is served from. See cpp/build-wasm.sh. */
const WASM_URL = '/ggml/zerotts-wasm.js';

export interface GgmlTimings {
  /** Text encode + cross-K/V precompute + [voice | soa] prefix, ms. */
  beginMs: number;
  /** Local depth transformer + sampling, ms — the hot path. */
  frameMs: number;
  /** Global decoder step, ms. */
  advanceMs: number;
  frames: number;
}

export class ZeroTTSGgml {
  readonly numCodebooks: number;
  readonly codebookSize: number;
  readonly nVoiceQueries: number;
  readonly dModel: number;
  readonly maxFrames: number;

  // Staging buffers in the WASM heap. Allocated once: the frame loop runs up to
  // 1500 times and these are only a few hundred bytes each.
  private readonly auPtr: number;
  private readonly codePtr: number;
  private readonly eoaPtr: number;
  private readonly timePtr: number;

  private constructor(
    private readonly M: ZeroTTSWasm,
    private readonly ctx: number,
    readonly tokenizer: BpeTokenizer,
  ) {
    this.numCodebooks  = M._zt_hparam(ctx, HP.NUM_CODEBOOKS);
    this.codebookSize  = M._zt_hparam(ctx, HP.CODEBOOK_SIZE);
    this.nVoiceQueries = M._zt_hparam(ctx, HP.N_VOICE_QUERIES);
    this.dModel        = M._zt_hparam(ctx, HP.D_MODEL);
    this.maxFrames     = M._zt_hparam(ctx, HP.MAX_FRAMES);

    this.auPtr   = M._zt_alloc(this.numCodebooks * 4);
    this.codePtr = M._zt_alloc(this.numCodebooks * 4);
    this.eoaPtr  = M._zt_alloc(4);
    this.timePtr = M._zt_alloc(4 * 8);
  }

  /**
   * @param gguf   the model file's bytes
   * @param threads worker threads for ggml; defaults to hardwareConcurrency
   *   capped at 8. Needs cross-origin isolation for SharedArrayBuffer, the same
   *   requirement onnxruntime-web's multi-threaded WASM already imposes.
   */
  static async create(
    gguf: ArrayBuffer, tokenizer: BpeTokenizer, threads?: number,
  ): Promise<ZeroTTSGgml> {
    // Imported through a constructed function so the bundler never sees this as
    // an import site. Emscripten's output is a pre-built ES module that resolves
    // its .wasm and spawns its pthread workers from its own URL; Vite's import
    // analysis rewrites that URL and then fails to transform the file, so the
    // module has to be fetched exactly as served from public/.
    const dynamicImport = new Function('u', 'return import(u)') as
      (u: string) => Promise<{ default: () => Promise<ZeroTTSWasm> }>;
    const factory = (await dynamicImport(WASM_URL)).default;
    const M: ZeroTTSWasm = await factory();

    const n = threads ?? Math.min(navigator.hardwareConcurrency || 4, 8);

    // The whole file has to be in the WASM heap for gguf to parse it, and the
    // weights are copied out of it into the backend buffer during load — so the
    // staging copy is freed immediately rather than held for the session.
    const bytes = new Uint8Array(gguf);
    const stage = M._zt_alloc(bytes.length);
    if (!stage) throw new Error('ggml: cannot allocate a staging buffer for the model');
    M.HEAPU8.set(bytes, stage);
    const ctx = M._zt_load(stage, bytes.length, n);
    M._zt_dealloc(stage);
    if (!ctx) throw new Error('ggml: failed to load the GGUF model');

    return new ZeroTTSGgml(M, ctx, tokenizer);
  }

  free(): void {
    this.M._zt_free(this.ctx);
  }

  timings(): GgmlTimings {
    this.M._zt_timings(this.ctx, this.timePtr);
    const t = new Float64Array(this.M.HEAPU8.buffer, this.timePtr, 4);
    return { beginMs: t[0], frameMs: t[1], advanceMs: t[2], frames: t[3] };
  }

  resetTimings(): void {
    this.M._zt_reset_timings(this.ctx);
  }

  /**
   * Yields one frame of codes at a time. Mirrors
   * `ZeroTTSBrowser.generateFrames` — same options, same draw order (one
   * control uniform then K audio uniforms per frame), same stop rule — so the
   * two backends are interchangeable and comparable.
   *
   * Codes come back as `Int32Array`; the ONNX path yields `BigInt64Array`
   * because ORT demands int64 there. Callers that pack frames for the codec
   * want int32 anyway.
   */
  async *generateFrames(
    text: string, voiceEmb: Float32Array, options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal, sharedRng?: Rng,
  ): AsyncGenerator<Int32Array> {
    const opts = { ...DEFAULT_SAMPLING, ...options };
    if (opts.cfgScale > 1.0) {
      throw new Error('the ggml backend does not implement CFG; use cfgScale = 1 ' +
                      'or the onnx backend');
    }
    if (voiceEmb.length !== this.nVoiceQueries * this.dModel) {
      throw new Error(`voice embedding has ${voiceEmb.length} floats, expected ` +
                      `${this.nVoiceQueries * this.dModel}`);
    }

    const { M, ctx } = this;
    const K = this.numCodebooks;
    const rng = sharedRng ?? new Rng(seed);

    // The tokenizer yields int64 because that is what the ONNX graphs demand;
    // ggml takes int32 ids, and the vocabulary is 8192 entries.
    const ids = Int32Array.from(this.tokenizer.encode(text), Number);
    const idsPtr = M._zt_alloc(ids.length * 4);
    const voicePtr = M._zt_alloc(voiceEmb.length * 4);
    try {
      M.HEAP32.set(ids, idsPtr >> 2);
      M.HEAPF32.set(voiceEmb, voicePtr >> 2);
      if (M._zt_begin(ctx, idsPtr, ids.length, voicePtr) !== 0) {
        throw new Error('ggml: zt_begin failed');
      }
    } finally {
      M._zt_dealloc(idsPtr);
      M._zt_dealloc(voicePtr);
    }

    const au = new Float32Array(K);
    let tailLeft: number | null = null;

    for (let t = 0; ; t++) {
      if (signal?.aborted) return;

      const forbidEoa = (t < opts.minFrames || tailLeft !== null) ? 1 : 0;
      const ctrlU = rng.next();
      rng.fill(au);
      M.HEAPF32.set(au, this.auPtr >> 2);

      if (M._zt_frame(ctx, forbidEoa, opts.textTemperature, opts.textTopK,
                      opts.audioTemperature, opts.audioTopK, opts.audioTopP,
                      opts.audioRepetitionPenalty, ctrlU, this.auPtr,
                      this.codePtr, this.eoaPtr) !== 0) {
        throw new Error(`ggml: zt_frame failed at frame ${t}`);
      }
      const isEoa = M.HEAP32[this.eoaPtr >> 2] !== 0;
      // Copied, not a view: HEAP32 is detached and replaced whenever the WASM
      // memory grows, and the caller keeps these past the next call.
      const codes = M.HEAP32.slice(this.codePtr >> 2, (this.codePtr >> 2) + K);

      if (tailLeft === null && isEoa) tailLeft = Math.max(0, opts.eoaExtraFrames);
      if ((tailLeft !== null && tailLeft <= 0) || t >= opts.maxFrames) return;

      yield codes;

      if (tailLeft !== null) {
        tailLeft -= 1;
        // Don't pay for a step whose output would be discarded.
        if (tailLeft <= 0) return;
      }

      if (M._zt_advance(ctx, this.codePtr, t) !== 0) {
        throw new Error(`ggml: zt_advance failed at frame ${t}`);
      }
    }
  }
}
