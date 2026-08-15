/**
 * MOSS-Audio-Tokenizer-Nano decoder — codes to waveform.
 *
 * Port of zerotts/codec.py. Decoder only; the encoder graph is not shipped
 * (it exists for voice cloning, which this release does not do).
 *
 * Three things differ from the TTS graphs and cause most porting bugs:
 *   - every integer tensor here is int32, NOT int64
 *   - the code axis order is (B, T, K) — time-major, codebook-last — while the
 *     TTS loop produces (B, K, T), so decode transposes
 *   - the streaming position ring buffer initializes to -1, not 0: position 0 is
 *     a real position, so a zero-filled buffer reads as "every slot holds
 *     frame 0"
 *
 * Credit: OpenMOSS team, Apache-2.0. See NOTICE.
 */

import * as ort from 'onnxruntime-web';

interface TransformerOffsetSpec {
  input_name: string;
  output_name: string;
  shape: number[];
}

interface AttentionCacheSpec {
  offset_input_name: string;
  offset_output_name: string;
  cached_keys_input_name: string;
  cached_keys_output_name: string;
  cached_values_input_name: string;
  cached_values_output_name: string;
  cached_positions_input_name: string;
  cached_positions_output_name: string;
  offset_shape: number[];
  cache_shape: number[];
  positions_shape: number[];
}

export interface CodecMeta {
  files: Record<string, string>;
  codec_config: {
    sample_rate: number;
    channels: number;
    downsample_rate: number;
    num_quantizers: number;
  };
  streaming_decode?: {
    transformer_offsets: TransformerOffsetSpec[];
    attention_caches: AttentionCacheSpec[];
  };
}

const prod = (dims: number[]) => dims.reduce((a, b) => a * b, 1);

/** (B, K, T) -> (B, T, K) int32, the layout the codec graphs expect. */
function transposeToBTK(codes: Int32Array, B: number, K: number, T: number): Int32Array {
  const out = new Int32Array(B * T * K);
  for (let b = 0; b < B; b++) {
    for (let k = 0; k < K; k++) {
      for (let t = 0; t < T; t++) {
        out[b * T * K + t * K + k] = codes[b * K * T + k * T + t];
      }
    }
  }
  return out;
}

/** (B, C, T) float32 -> (B, T) mono by averaging channels. */
function toMono(data: Float32Array, B: number, C: number, T: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    let sum = 0;
    for (let c = 0; c < C; c++) sum += data[c * T + t];
    out[t] = sum / C;
  }
  return out;
}

export class MossCodecDecoder {
  readonly sampleRate: number;
  readonly numChannels: number;
  readonly numCodebooks: number;
  readonly frameSize: number;

  private constructor(
    private decodeFull: ort.InferenceSession,
    private decodeStep: ort.InferenceSession,
    private meta: CodecMeta,
  ) {
    const cfg = meta.codec_config;
    this.sampleRate = cfg.sample_rate;
    this.numChannels = cfg.channels;
    this.numCodebooks = cfg.num_quantizers;
    this.frameSize = cfg.downsample_rate;
  }

  static async create(
    meta: CodecMeta,
    graphs: { decodeFull: ArrayBuffer; decodeStep: ArrayBuffer },
    options: ort.InferenceSession.SessionOptions,
  ): Promise<MossCodecDecoder> {
    const [full, step] = await Promise.all([
      ort.InferenceSession.create(graphs.decodeFull, options),
      ort.InferenceSession.create(graphs.decodeStep, options),
    ]);
    return new MossCodecDecoder(full, step, meta);
  }

  /** (K, T) codes -> mono float32 at sampleRate. One call for the utterance. */
  async decode(codesKT: Int32Array, K: number, T: number): Promise<Float32Array> {
    const btk = transposeToBTK(codesKT, 1, K, T);
    const outputs = await this.decodeFull.run({
      audio_codes: new ort.Tensor('int32', btk, [1, T, K]),
      audio_code_lengths: new ort.Tensor('int32', Int32Array.from([T]), [1]),
    });
    const audio = outputs.audio as ort.Tensor;
    const lengths = outputs.audio_lengths as ort.Tensor;
    const n = Number((lengths.data as Int32Array)[0]);
    const [, C, Tfull] = audio.dims as number[];
    return toMono(audio.data as Float32Array, 1, C, Tfull, n);
  }

  streamingDecoder(): MossStreamingDecoder {
    return new MossStreamingDecoder(this.decodeStep, this.meta);
  }
}

export class MossStreamingDecoder {
  private state = new Map<string, ort.Tensor>();
  private transformerSpecs: TransformerOffsetSpec[];
  private attentionSpecs: AttentionCacheSpec[];

  constructor(private session: ort.InferenceSession, meta: CodecMeta) {
    const streaming = meta.streaming_decode ?? { transformer_offsets: [], attention_caches: [] };
    this.transformerSpecs = streaming.transformer_offsets;
    this.attentionSpecs = streaming.attention_caches;
    this.reset();
  }

  reset(): void {
    this.state.clear();
    for (const spec of this.transformerSpecs) {
      this.state.set(spec.input_name,
        new ort.Tensor('int32', new Int32Array(prod(spec.shape)), spec.shape));
    }
    for (const spec of this.attentionSpecs) {
      this.state.set(spec.offset_input_name,
        new ort.Tensor('int32', new Int32Array(prod(spec.offset_shape)), spec.offset_shape));
      this.state.set(spec.cached_keys_input_name,
        new ort.Tensor('float32', new Float32Array(prod(spec.cache_shape)), spec.cache_shape));
      this.state.set(spec.cached_values_input_name,
        new ort.Tensor('float32', new Float32Array(prod(spec.cache_shape)), spec.cache_shape));
      // -1, not 0 — see the file header.
      const positions = new Int32Array(prod(spec.positions_shape)).fill(-1);
      this.state.set(spec.cached_positions_input_name,
        new ort.Tensor('int32', positions, spec.positions_shape));
    }
  }

  /** (K, frames) codes for this chunk -> mono float32. */
  async decodeChunk(codesKT: Int32Array, K: number, frames: number): Promise<Float32Array> {
    const btk = transposeToBTK(codesKT, 1, K, frames);
    const feeds: Record<string, ort.Tensor> = {
      audio_codes: new ort.Tensor('int32', btk, [1, frames, K]),
      audio_code_lengths: new ort.Tensor('int32', Int32Array.from([frames]), [1]),
    };
    for (const [k, v] of this.state) feeds[k] = v;

    const outputs = await this.session.run(feeds);

    for (const spec of this.transformerSpecs) {
      this.state.set(spec.input_name, outputs[spec.output_name] as ort.Tensor);
    }
    for (const spec of this.attentionSpecs) {
      this.state.set(spec.offset_input_name, outputs[spec.offset_output_name] as ort.Tensor);
      this.state.set(spec.cached_keys_input_name, outputs[spec.cached_keys_output_name] as ort.Tensor);
      this.state.set(spec.cached_values_input_name, outputs[spec.cached_values_output_name] as ort.Tensor);
      this.state.set(spec.cached_positions_input_name, outputs[spec.cached_positions_output_name] as ort.Tensor);
    }

    const audio = outputs.audio as ort.Tensor;
    const lengths = (outputs.audio_lengths as ort.Tensor).data as Int32Array;
    const n = Number(lengths[0]);
    const [, C, T] = audio.dims as number[];
    return toMono(audio.data as Float32Array, 1, C, T, n);
  }
}
