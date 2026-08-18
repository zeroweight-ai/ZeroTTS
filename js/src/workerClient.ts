/**
 * Main-thread handle on the synthesis worker (see worker.ts for why there is
 * one). Turns the message port into promises and an async iterator, so the UI
 * code reads the same as it did when the model ran inline.
 *
 * The worker module is NOT imported statically anywhere on the page: the
 * `new Worker(new URL(...))` form is what lets Vite build it as its own bundle,
 * which is what keeps onnxruntime-web off the UI thread's dependency graph.
 */

import { DownloadProgress } from './cache';
import {
  GenerateParams, LoadedInfo, WorkerRequest, WorkerResponse,
} from './workerProtocol';

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onProgress?: (p: DownloadProgress) => void;
  onChunk?: (chunk: Float32Array) => void;
}

export class TtsWorker {
  private worker = new Worker(new URL('./worker.ts', import.meta.url), {
    type: 'module',
    name: 'zerotts',
  });

  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      const entry = this.pending.get(message.id);
      if (!entry) return;
      switch (message.type) {
        case 'progress': entry.onProgress?.(message.progress); break;
        case 'chunk': entry.onChunk?.(message.chunk); break;
        case 'result':
          this.pending.delete(message.id);
          entry.resolve(message.value);
          break;
        case 'error':
          this.pending.delete(message.id);
          entry.reject(new Error(message.message));
          break;
      }
    };
    // A worker that dies (OOM on a 900 MB model is the realistic cause) would
    // otherwise leave every caller awaiting forever.
    this.worker.onerror = (event) => this.failAll(event.message || 'worker crashed');
    this.worker.onmessageerror = () => this.failAll('worker sent an unreadable message');
  }

  private failAll(message: string): void {
    for (const [, entry] of this.pending) entry.reject(new Error(message));
    this.pending.clear();
  }

  private request<T>(
    message: WorkerRequest, hooks: Omit<Pending, 'resolve' | 'reject'> = {},
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.id, { resolve: resolve as Pending['resolve'], reject, ...hooks });
      this.worker.postMessage(message);
    });
  }

  downloadInfo(repo: string): Promise<{ bytes: number; cached: boolean }> {
    return this.request({ type: 'downloadInfo', id: this.nextId++, repo });
  }

  clearCache(): Promise<void> {
    return this.request({ type: 'clearCache', id: this.nextId++ });
  }

  load(repo: string, onProgress?: (p: DownloadProgress) => void): Promise<LoadedInfo> {
    return this.request({ type: 'load', id: this.nextId++, repo }, { onProgress });
  }

  /**
   * Stream a take. Chunks arrive as they are decoded; `cancel()` on the returned
   * handle stops the worker's loop at the next frame boundary.
   */
  generate(params: GenerateParams): { chunks: AsyncIterable<Float32Array>; cancel: () => void } {
    const id = this.nextId++;

    // A queue rather than a callback: the consumer (`for await`) may be slower
    // than the worker, and chunks must not be dropped when it is.
    const queue: Float32Array[] = [];
    // A plain object, not locals: these are written from callbacks, and the
    // narrowing TypeScript applies to a `let` read inside the generator would be
    // wrong about both of them.
    const state = { finished: false, failure: null as Error | null };
    let notify: (() => void) | null = null;

    const wake = () => { notify?.(); notify = null; };

    this.request(
      { type: 'generate', id, params },
      { onChunk: (chunk) => { queue.push(chunk); wake(); } },
    ).then(
      () => { state.finished = true; wake(); },
      (error: Error) => { state.failure = error; state.finished = true; wake(); },
    );

    const chunks: AsyncIterable<Float32Array> = {
      async *[Symbol.asyncIterator]() {
        for (;;) {
          while (queue.length) yield queue.shift() as Float32Array;
          if (state.failure) throw state.failure;
          if (state.finished) return;
          await new Promise<void>((resolve) => { notify = resolve; });
        }
      },
    };

    return {
      chunks,
      cancel: () => this.worker.postMessage(
        { type: 'cancel', id: this.nextId++, target: id } satisfies WorkerRequest),
    };
  }
}
