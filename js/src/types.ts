/** Shared types — mirrors config.json and voices/index.json in the weights repo. */

export interface ZeroTTSConfig {
  text_format: string;
  vocab_size: number;
  num_codebooks: number;
  codebook_size: number;
  d_model: number;
  n_heads: number;
  n_layers: number;
  n_voice_queries: number;
  special_tokens: Record<string, number>;
  sample_rate: number;
  codec_frame_rate: number;
}

export interface VoiceInfo {
  name: string;
  language: string;
  description: string;
  n_voice_queries: number;
  shape: number[];
}

export interface VoiceIndex {
  voices: VoiceInfo[];
}

export interface SamplingOptions {
  cfgScale: number;
  textTemperature: number;
  textTopK: number;
  audioTemperature: number;
  audioTopK: number;
  audioTopP: number;
  audioRepetitionPenalty: number;
  minFrames: number;
  maxFrames: number;
  /** Frames kept at/after the stop signal. See docs/RUNTIME.md. */
  eoaExtraFrames: number;
}

export const DEFAULT_SAMPLING: SamplingOptions = {
  cfgScale: 1.0,
  textTemperature: 1.0,
  textTopK: 50,
  audioTemperature: 0.8,
  audioTopK: 25,
  audioTopP: 0.95,
  audioRepetitionPenalty: 1.2,
  minFrames: 4,
  maxFrames: 1500,
  eoaExtraFrames: 1,
};

export interface ProgressEvent {
  phase: 'download' | 'init' | 'generate' | 'done' | 'error';
  message: string;
  /** 0..1 where known. */
  fraction?: number;
}
