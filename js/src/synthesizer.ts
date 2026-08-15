/**
 * ZeroTTS in the browser — port of zerotts/synthesizer.py.
 *
 * Same contract as the Python runtime (docs/RUNTIME.md): two ONNX Runtime calls
 * per audio frame, frames at 12.5 Hz, decoded by the bundled MOSS codec.
 *
 * Notes specific to this port:
 *   - the TTS graphs take int64 (BigInt64Array here); the codec takes int32
 *   - `seenMask` is a (1, K, codebookSize) bool array mutated in place every
 *     frame; it is allocated once and never reallocated
 *   - random draws are supplied by a seedable Rng (see rng.ts), which is what
 *     makes this port testable against Python for bit-exact agreement
 */

import * as ort from 'onnxruntime-web';

import { MossCodecDecoder, MossStreamingDecoder } from './codec';
import { Rng } from './rng';
import { BpeTokenizer } from './tokenizer';
import { DEFAULT_SAMPLING, SamplingOptions, ZeroTTSConfig } from './types';

const big = (values: ArrayLike<number>) =>
  BigInt64Array.from(Array.from(values, (v) => BigInt(v)));

export interface Sessions {
  textEncoder: ort.InferenceSession;
  prefixStep: ort.InferenceSession;
  localFrameDecode: ort.InferenceSession;
}

interface PrefixState {
  hidden: ort.Tensor;       // (B, D) — predicts the next frame
  packedKv: ort.Tensor;
  fullValid: ort.Tensor;
  textStates: ort.Tensor;
  textValid: ort.Tensor;
}

export class ZeroTTSBrowser {
  readonly sampleRate: number;
  readonly numCodebooks: number;
  readonly nVoiceQueries: number;
  private readonly dModel: number;
  private readonly nLayers: number;
  private readonly nHeads: number;
  private readonly dHead: number;
  private readonly codebookSize: number;

  constructor(
    private sessions: Sessions,
    private codec: MossCodecDecoder,
    private tokenizer: BpeTokenizer,
    private config: ZeroTTSConfig,
    private nullVoiceEmb: Float32Array,
  ) {
    this.sampleRate = config.sample_rate;
    this.numCodebooks = config.num_codebooks;
    this.codebookSize = config.codebook_size;
    this.dModel = config.d_model;
    this.nLayers = config.n_layers;
    this.nHeads = config.n_heads;
    this.dHead = config.d_model / config.n_heads;
    this.nVoiceQueries = config.n_voice_queries;
  }

  /** Push one dummy request through the hot-path sessions so first-call latency
   *  reflects steady state rather than lazy allocator setup. */
  async warmup(): Promise<void> {
    const rng = new Rng(0);
    const state = await this.prefixStepInit(big([1, 2]), 2, this.nullVoiceEmb, 1);
    const seen = new Uint8Array(this.numCodebooks * this.codebookSize);
    const { codes } = await this.localDecodeFrame(
      state.hidden, true, DEFAULT_SAMPLING, seen, 1, rng);
    await this.prefixStepFrame(codes, 0, state, this.nVoiceQueries, 1);
  }

  // ── graph calls ────────────────────────────────────────────────────────────

  private async prefixStepInit(
    textIds: BigInt64Array, textLen: number, voiceEmb: Float32Array, B: number,
  ): Promise<PrefixState> {
    const V = this.nVoiceQueries;
    const D = this.dModel;

    // Text ids are broadcast across the guidance batch — only the voice differs.
    const idsBatched = B === 1 ? textIds : (() => {
      const out = new BigInt64Array(B * textLen);
      for (let b = 0; b < B; b++) out.set(textIds, b * textLen);
      return out;
    })();

    const encoded = await this.sessions.textEncoder.run({
      text_ids: new ort.Tensor('int64', idsBatched, [B, textLen]),
      txt_lengths: new ort.Tensor('int64', big(new Array(B).fill(textLen)), [B]),
    });
    const textStates = encoded.text_states as ort.Tensor;
    const textValid = encoded.text_valid as ort.Tensor;
    const soaEmbed = encoded.soa_embed as ort.Tensor;

    // external_embed = [voice | soa]
    const T = V + 1;
    const external = new Float32Array(B * T * D);
    const soa = soaEmbed.data as Float32Array;
    for (let b = 0; b < B; b++) {
      external.set(voiceEmb.subarray(b * V * D, (b + 1) * V * D), b * T * D);
      external.set(soa.subarray(b * D, (b + 1) * D), b * T * D + V * D);
    }

    const newPos = new BigInt64Array(B * T);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) newPos[b * T + t] = BigInt(t);

    const bidirectional = new Uint8Array(B * T);
    for (let b = 0; b < B; b++) bidirectional.fill(1, b * T, b * T + V);

    const outputs = await this.sessions.prefixStep.run({
      external_embed: new ort.Tensor('float32', external, [B, T, D]),
      use_external_embed: new ort.Tensor('bool', new Uint8Array(B * T).fill(1), [B, T]),
      frame_codes: new ort.Tensor('int64', new BigInt64Array(B * T * this.numCodebooks),
        [B, T, this.numCodebooks]),
      new_pos: new ort.Tensor('int64', newPos, [B, T]),
      new_valid: new ort.Tensor('bool', new Uint8Array(B * T).fill(1), [B, T]),
      packed_kv: new ort.Tensor('float32', new Float32Array(0),
        [this.nLayers, 2, B, this.nHeads, 0, this.dHead]),
      new_bidirectional: new ort.Tensor('bool', bidirectional, [B, T]),
      past_valid: new ort.Tensor('bool', new Uint8Array(0), [B, 0]),
      text_states: textStates,
      text_valid: textValid,
    });

    return {
      hidden: lastPosition(outputs.hidden as ort.Tensor),
      packedKv: outputs.packed_kv as ort.Tensor,
      fullValid: outputs.full_valid as ort.Tensor,
      textStates, textValid,
    };
  }

  private async prefixStepFrame(
    codes: BigInt64Array, frameIndex: number, state: PrefixState, nVoice: number, B: number,
  ): Promise<PrefixState> {
    const K = this.numCodebooks;

    // One sampled frame is the history BOTH guidance branches continue from.
    const tiled = B === 1 ? codes : (() => {
      const out = new BigInt64Array(B * K);
      for (let b = 0; b < B; b++) out.set(codes, b * K);
      return out;
    })();

    // Position: the voice block holds 0..V-1, <soa> is at V, frame t at V+1+t.
    const pos = new BigInt64Array(B).fill(BigInt(nVoice + 1 + frameIndex));

    const outputs = await this.sessions.prefixStep.run({
      external_embed: new ort.Tensor('float32', new Float32Array(B * this.dModel),
        [B, 1, this.dModel]),
      use_external_embed: new ort.Tensor('bool', new Uint8Array(B), [B, 1]),
      frame_codes: new ort.Tensor('int64', tiled, [B, 1, K]),
      new_pos: new ort.Tensor('int64', pos, [B, 1]),
      new_valid: new ort.Tensor('bool', new Uint8Array(B).fill(1), [B, 1]),
      packed_kv: state.packedKv,
      past_valid: state.fullValid,
      text_states: state.textStates,
      new_bidirectional: new ort.Tensor('bool', new Uint8Array(B), [B, 1]),
      text_valid: state.textValid,
    });

    return {
      hidden: lastPosition(outputs.hidden as ort.Tensor),
      packedKv: outputs.packed_kv as ort.Tensor,
      fullValid: outputs.full_valid as ort.Tensor,
      textStates: state.textStates,
      textValid: state.textValid,
    };
  }

  private async localDecodeFrame(
    hidden: ort.Tensor, forbidEoa: boolean, opts: SamplingOptions,
    seenMask: Uint8Array, B: number, rng: Rng,
  ): Promise<{ isEoa: boolean; codes: BigInt64Array }> {
    const K = this.numCodebooks;
    const outputs = await this.sessions.localFrameDecode.run({
      global_hidden: hidden,
      forbid_eoa: new ort.Tensor('bool', Uint8Array.from([forbidEoa ? 1 : 0]), [1]),
      text_temperature: new ort.Tensor('float32', Float32Array.from([opts.textTemperature]), [1]),
      text_topk: new ort.Tensor('int64', big([opts.textTopK > 0 ? opts.textTopK : this.codebookSize]), [1]),
      audio_temperature: new ort.Tensor('float32', Float32Array.from([opts.audioTemperature]), [1]),
      audio_topk: new ort.Tensor('int64', big([opts.audioTopK > 0 ? opts.audioTopK : this.codebookSize]), [1]),
      audio_topp: new ort.Tensor('float32', Float32Array.from([opts.audioTopP]), [1]),
      audio_repetition_penalty: new ort.Tensor('float32',
        Float32Array.from([opts.audioRepetitionPenalty]), [1]),
      seen_mask: new ort.Tensor('bool', seenMask, [1, K, this.codebookSize]),
      ctrl_random_u: new ort.Tensor('float32', rng.fill(new Float32Array(1)), [1]),
      audio_random_u: new ort.Tensor('float32', rng.fill(new Float32Array(K)), [1, K]),
      cfg_scale: new ort.Tensor('float32', Float32Array.from([opts.cfgScale]), [1]),
    });

    const codes = outputs.codes.data as BigInt64Array;
    // The graph cannot carry a variable-length history, so the caller maintains
    // the repetition-penalty mask.
    for (let c = 0; c < K; c++) {
      seenMask[c * this.codebookSize + Number(codes[c])] = 1;
    }
    return { isEoa: Boolean((outputs.is_eoa.data as Uint8Array)[0]), codes };
  }

  // ── generation ─────────────────────────────────────────────────────────────

  /** Yields one frame of codes (length numCodebooks) at a time. */
  async *generateFrames(
    text: string, voiceEmb: Float32Array | null, options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal,
  ): AsyncGenerator<BigInt64Array> {
    const opts = { ...DEFAULT_SAMPLING, ...options };
    const rng = new Rng(seed);
    let voice = voiceEmb ?? this.nullVoiceEmb;
    const B = opts.cfgScale > 1.0 ? 2 : 1;
    if (B === 2) {
      const stacked = new Float32Array(voice.length + this.nullVoiceEmb.length);
      stacked.set(voice, 0);
      stacked.set(this.nullVoiceEmb, voice.length);
      voice = stacked;
    }

    const ids = this.tokenizer.encode(text);
    let state = await this.prefixStepInit(ids, ids.length, voice, B);

    const seenMask = new Uint8Array(this.numCodebooks * this.codebookSize);
    let tailLeft: number | null = null;

    for (let t = 0; ; t++) {
      if (signal?.aborted) return;

      const { isEoa, codes } = await this.localDecodeFrame(
        state.hidden, t < opts.minFrames || tailLeft !== null, opts, seenMask, B, rng);

      if (tailLeft === null && isEoa) tailLeft = Math.max(0, opts.eoaExtraFrames);
      if ((tailLeft !== null && tailLeft <= 0) || t >= opts.maxFrames) return;

      yield codes;

      if (tailLeft !== null) {
        tailLeft -= 1;
        // Don't pay for a step whose output would be discarded.
        if (tailLeft <= 0) return;
      }

      state = await this.prefixStepFrame(codes, t, state, this.nVoiceQueries, B);
    }
  }

  /** Generate the whole utterance, then decode once. */
  async synthesize(
    text: string, voiceEmb: Float32Array | null, options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal,
  ): Promise<Float32Array> {
    const frames: BigInt64Array[] = [];
    for await (const f of this.generateFrames(text, voiceEmb, options, seed, signal)) {
      frames.push(f);
    }
    if (!frames.length) return new Float32Array(0);
    return this.codec.decode(...packFrames(frames, this.numCodebooks));
  }

  /**
   * Streaming: yields audio chunks as they are produced. The chunk size ramps —
   * the first is one frame so playback starts fast, then it doubles up to 16 so
   * the per-call codec overhead is amortized once a buffer exists.
   */
  async *synthesizeStream(
    text: string, voiceEmb: Float32Array | null, options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal,
    firstChunkFrames = 1, maxChunkFrames = 16,
  ): AsyncGenerator<Float32Array> {
    const stream: MossStreamingDecoder = this.codec.streamingDecoder();
    let target = Math.max(1, firstChunkFrames);
    const cap = Math.max(target, maxChunkFrames);
    let buffer: BigInt64Array[] = [];

    const flush = async () => {
      const [codes, K, n] = packFrames(buffer, this.numCodebooks);
      buffer = [];
      return stream.decodeChunk(codes, K, n);
    };

    for await (const frame of this.generateFrames(text, voiceEmb, options, seed, signal)) {
      buffer.push(frame);
      if (buffer.length >= target) {
        yield await flush();
        target = Math.min(cap, target * 2);
      }
    }
    if (buffer.length) yield await flush();
  }
}

/** Frames (each length K) -> (K, T) int32, the layout the codec wants. */
function packFrames(frames: BigInt64Array[], K: number): [Int32Array, number, number] {
  const T = frames.length;
  const out = new Int32Array(K * T);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) out[k * T + t] = Number(frames[t][k]);
  }
  return [out, K, T];
}

/** hidden is (B, T, D); the runtime only ever wants the last position. */
function lastPosition(hidden: ort.Tensor): ort.Tensor {
  const [B, T, D] = hidden.dims as number[];
  if (T === 1) return new ort.Tensor('float32', hidden.data as Float32Array, [B, D]);
  const data = hidden.data as Float32Array;
  const out = new Float32Array(B * D);
  for (let b = 0; b < B; b++) {
    out.set(data.subarray(b * T * D + (T - 1) * D, b * T * D + T * D), b * D);
  }
  return new ort.Tensor('float32', out, [B, D]);
}
