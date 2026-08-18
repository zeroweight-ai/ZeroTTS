/**
 * The messages exchanged with the synthesis worker.
 *
 * Shared by both sides so a renamed field is a type error rather than a message
 * that is silently ignored at runtime.
 */

import { DownloadProgress } from './cache';
import { SamplingOptions } from './types';
import { VoiceIndex } from './types';

export interface GenerateParams {
  segments: string[];
  voiceName: string;
  options: Partial<SamplingOptions>;
  seed?: number;
}

export interface LoadedInfo {
  voices: VoiceIndex;
  base: string;
  sampleRate: number;
}

export type WorkerRequest =
  | { type: 'downloadInfo'; id: number; repo: string }
  | { type: 'clearCache'; id: number }
  | { type: 'load'; id: number; repo: string }
  | { type: 'generate'; id: number; params: GenerateParams }
  /** Cancels the in-flight `generate` whose id is `target`. */
  | { type: 'cancel'; id: number; target: number };

export type WorkerResponse =
  | { type: 'result'; id: number; value: unknown }
  | { type: 'error'; id: number; message: string }
  | { type: 'progress'; id: number; progress: DownloadProgress }
  | { type: 'chunk'; id: number; chunk: Float32Array };
