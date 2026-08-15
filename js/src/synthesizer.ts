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

/**
 * Coerce anything integer-ish into a real BigInt64Array.
 *
 * ONNX Runtime rejects an int64 tensor whose data is not literally a
 * `BigInt64Array` ("A int64 tensor's data must be type of function
 * BigInt64Array()"). Two things make that easy to hit:
 *
 *  - a graph's int64 OUTPUT is not guaranteed to come back as a BigInt64Array
 *    across ORT-web versions and execution providers (WebGPU in particular has
 *    no native int64), and `outputs.x.data as BigInt64Array` is a TypeScript
 *    cast that asserts rather than checks — so a wrong type survives until the
 *    value is fed back in as an input;
 *  - a typed array from another realm fails `instanceof`.
 *
 * Going through this helper on every int64 input makes the runtime behaviour
 * match the type annotations instead of merely claiming to.
 */
function toBigInt64(values: ArrayLike<number | bigint>): BigInt64Array {
  if (values instanceof BigInt64Array) return values;
  const out = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = typeof v === 'bigint' ? v : BigInt(Math.trunc(v as number));
  }
  return out;
}

/** An int64 tensor, from any integer-ish source. */
const i64 = (values: ArrayLike<number | bigint>, dims: number[]) =>
  new ort.Tensor('int64', toBigInt64(values), dims);

const big = (values: ArrayLike<number>) => toBigInt64(values);

/**
 * Run a session, then return its outputs BY POSITION.
 *
 * The Python runtime reads outputs positionally (`h, kv, valid = sess.run(...)`)
 * and so never depends on their names. Reading them by name here reintroduced
 * that coupling and broke on the first mismatch: prefix_step's KV output is
 * called `new_packed_kv`, not `packed_kv`, so `outputs.packed_kv` was undefined
 * and the *next* call failed with "input 'packed_kv' is missing in 'feeds'" —
 * an error pointing at the input, one call after the actual mistake.
 *
 * Feeds are checked too: ORT reports an undefined value as a missing input,
 * which reads as "you forgot this" when the truth is "the value you passed was
 * undefined". Failing here names the culprit.
 */
async function run(
  session: ort.InferenceSession, feeds: Record<string, ort.Tensor>,
): Promise<ort.Tensor[]> {
  for (const [name, value] of Object.entries(feeds)) {
    if (value == null) {
      throw new Error(
        `feed '${name}' is ${value} — an earlier graph output was probably read `
        + `under the wrong name (this session's outputs are: `
        + `${session.outputNames.join(', ')})`);
    }
  }
  const outputs = await session.run(feeds);
  return session.outputNames.map((name) => outputs[name] as ort.Tensor);
}

/** Frames of codec silence decoded between text segments. */
const SILENCE_FRAMES_PER_SEGMENT = 4;

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
    /** The codec's canonical silence frame (length numCodebooks), or null when
     *  the model directory predates silence_frame.npy — see synthesizeStream. */
    private silenceFrame: BigInt64Array | null = null,
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

    const [textStates, textValid, soaEmbed] = await run(this.sessions.textEncoder, {
      text_ids: i64(idsBatched, [B, textLen]),
      txt_lengths: i64(new Array(B).fill(textLen), [B]),
    });

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

    const [hidden, packedKv, fullValid] = await run(this.sessions.prefixStep, {
      external_embed: new ort.Tensor('float32', external, [B, T, D]),
      use_external_embed: new ort.Tensor('bool', new Uint8Array(B * T).fill(1), [B, T]),
      frame_codes: i64(new BigInt64Array(B * T * this.numCodebooks),
        [B, T, this.numCodebooks]),
      new_pos: i64(newPos, [B, T]),
      new_valid: new ort.Tensor('bool', new Uint8Array(B * T).fill(1), [B, T]),
      packed_kv: new ort.Tensor('float32', new Float32Array(0),
        [this.nLayers, 2, B, this.nHeads, 0, this.dHead]),
      new_bidirectional: new ort.Tensor('bool', bidirectional, [B, T]),
      past_valid: new ort.Tensor('bool', new Uint8Array(0), [B, 0]),
      text_states: textStates,
      text_valid: textValid,
    });

    return {
      hidden: lastPosition(hidden), packedKv, fullValid, textStates, textValid,
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

    const [hidden, packedKv, fullValid] = await run(this.sessions.prefixStep, {
      external_embed: new ort.Tensor('float32', new Float32Array(B * this.dModel),
        [B, 1, this.dModel]),
      use_external_embed: new ort.Tensor('bool', new Uint8Array(B), [B, 1]),
      frame_codes: i64(tiled, [B, 1, K]),
      new_pos: i64(pos, [B, 1]),
      new_valid: new ort.Tensor('bool', new Uint8Array(B).fill(1), [B, 1]),
      packed_kv: state.packedKv,
      past_valid: state.fullValid,
      text_states: state.textStates,
      new_bidirectional: new ort.Tensor('bool', new Uint8Array(B), [B, 1]),
      text_valid: state.textValid,
    });

    return {
      hidden: lastPosition(hidden), packedKv, fullValid,
      textStates: state.textStates, textValid: state.textValid,
    };
  }

  private async localDecodeFrame(
    hidden: ort.Tensor, forbidEoa: boolean, opts: SamplingOptions,
    seenMask: Uint8Array, B: number, rng: Rng,
  ): Promise<{ isEoa: boolean; codes: BigInt64Array }> {
    const K = this.numCodebooks;
    const [isEoaT, codesT] = await run(this.sessions.localFrameDecode, {
      global_hidden: hidden,
      forbid_eoa: new ort.Tensor('bool', Uint8Array.from([forbidEoa ? 1 : 0]), [1]),
      text_temperature: new ort.Tensor('float32', Float32Array.from([opts.textTemperature]), [1]),
      text_topk: i64([opts.textTopK > 0 ? opts.textTopK : this.codebookSize], [1]),
      audio_temperature: new ort.Tensor('float32', Float32Array.from([opts.audioTemperature]), [1]),
      audio_topk: i64([opts.audioTopK > 0 ? opts.audioTopK : this.codebookSize], [1]),
      audio_topp: new ort.Tensor('float32', Float32Array.from([opts.audioTopP]), [1]),
      audio_repetition_penalty: new ort.Tensor('float32',
        Float32Array.from([opts.audioRepetitionPenalty]), [1]),
      seen_mask: new ort.Tensor('bool', seenMask, [1, K, this.codebookSize]),
      ctrl_random_u: new ort.Tensor('float32', rng.fill(new Float32Array(1)), [1]),
      audio_random_u: new ort.Tensor('float32', rng.fill(new Float32Array(K)), [1, K]),
      cfg_scale: new ort.Tensor('float32', Float32Array.from([opts.cfgScale]), [1]),
    });

    // Coerced, not cast: this value is fed straight back in as frame_codes.
    const codes = toBigInt64(codesT.data as ArrayLike<number | bigint>);
    // The graph cannot carry a variable-length history, so the caller maintains
    // the repetition-penalty mask.
    for (let c = 0; c < K; c++) {
      seenMask[c * this.codebookSize + Number(codes[c])] = 1;
    }
    return { isEoa: Boolean((isEoaT.data as Uint8Array)[0]), codes };
  }

  // ── generation ─────────────────────────────────────────────────────────────

  /** Yields one frame of codes (length numCodebooks) at a time. */
  async *generateFrames(
    text: string, voiceEmb: Float32Array | null, options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal,
    /** Supply one to continue a draw sequence across several segments. Creating
     *  a fresh Rng per segment from the same seed would replay the identical
     *  draws for every segment. */
    sharedRng?: Rng,
  ): AsyncGenerator<BigInt64Array> {
    const opts = { ...DEFAULT_SAMPLING, ...options };
    const rng = sharedRng ?? new Rng(seed);
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
   * Streaming synthesis over one or more text segments.
   *
   * Each segment is its own generate call — the architecture re-primes the
   * global transformer's [voice | soa] prefix fresh per call, so segments cannot
   * share that.
   *
   * The CODEC, however, is ONE continuous streaming session across every
   * segment. Its decoder is causal and KV-cached; opening a fresh decoder per
   * segment restarts that cache cold, and the discontinuity is audible as a
   * click at every segment boundary. This is the single easiest thing to get
   * wrong here, and it fails quietly — each segment on its own sounds correct.
   *
   * Between segments a few frames of the codec's canonical silence are decoded
   * through the same session, so the previous segment's last phone gets a breath
   * instead of butting straight against the next one's first. Those codes are
   * not zeros (code 0 is a real code, not silence) — they are shipped alongside
   * the weights as silence_frame.npy, because deriving them needs the codec
   * encoder, which is not part of this release. Without the file, padding is
   * skipped rather than guessed.
   *
   * The chunk-size ramp (1, 2, 4 … frames per decode call) only buys
   * time-to-first-audio, which only matters once, so it applies to the first
   * segment only; later segments decode straight at `maxChunkFrames`.
   */
  async *synthesizeStream(
    segments: string | string[], voiceEmb: Float32Array | null,
    options: Partial<SamplingOptions> = {},
    seed?: number, signal?: AbortSignal,
    firstChunkFrames = 1, maxChunkFrames = 16,
  ): AsyncGenerator<Float32Array> {
    const texts = Array.isArray(segments) ? segments : [segments];
    const stream: MossStreamingDecoder = this.codec.streamingDecoder();
    const cap = Math.max(1, maxChunkFrames);
    // ONE draw sequence across all segments: a fresh Rng per segment from the
    // same seed would give every segment identical random draws.
    const rng = new Rng(seed);
    const silence = this.silenceFrame;
    let buffer: BigInt64Array[] = [];

    const flush = async () => {
      const [codes, K, n] = packFrames(buffer, this.numCodebooks);
      buffer = [];
      return stream.decodeChunk(codes, K, n);
    };

    for (let segIdx = 0; segIdx < texts.length; segIdx++) {
      if (signal?.aborted) return;
      let target = segIdx === 0 ? Math.max(1, firstChunkFrames) : cap;

      for await (const frame of this.generateFrames(
        texts[segIdx], voiceEmb, options, seed, signal, rng,
      )) {
        buffer.push(frame);
        if (buffer.length >= target) {
          yield await flush();
          if (segIdx === 0) target = Math.min(cap, target * 2);
        }
      }
      if (buffer.length) yield await flush();

      if (silence) {
        buffer = Array.from({ length: SILENCE_FRAMES_PER_SEGMENT }, () => silence);
        yield await flush();
      }
    }
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
