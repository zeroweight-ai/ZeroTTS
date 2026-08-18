/**
 * The synthesis worker — everything that touches ONNX Runtime lives here.
 *
 * Why a worker at all: ORT-web's WASM backend runs inference on the thread that
 * calls it. `session.run()` returns a Promise, but the compute inside it is a
 * synchronous blocking call, and this model does TWO of them per 80 ms frame for
 * up to 1500 frames. On the UI thread that means the page cannot paint, scroll,
 * or even service the AudioWorklet's message port for the whole generation — the
 * tab looks hung. Off the UI thread, the same work is invisible: the page stays
 * responsive and Stop actually stops.
 *
 * The worker owns the sessions, the loaded voices and the generation loop; the
 * page only ever sees text in and Float32Array chunks out.
 */

import { clearCache } from './cache';
import { loadModel } from './loader';
import { DEFAULT_REPO, downloadInfo, loadVoice, repoBaseUrl } from './repo';
import { ZeroTTSBrowser } from './synthesizer';
import { GenerateParams, LoadedInfo, WorkerRequest, WorkerResponse } from './workerProtocol';

let tts: ZeroTTSBrowser | null = null;
let base = '';

/** Voice latents are a few MB and are re-used on every take. */
const voices = new Map<string, Float32Array>();
/** In-flight generations, so `cancel` can reach the right one. */
const running = new Map<number, AbortController>();

const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

/**
 * Give the worker's event loop one turn.
 *
 * `await session.run()` looks like it yields, but ORT's WASM compute is
 * synchronous and its promise resolves in a MICROTASK — so an `await` inside the
 * generation loop hands control straight back to the next frame and the message
 * queue is never drained. Without a real macrotask in the loop, `cancel` sits
 * unread until generation ends on its own, and Stop does nothing.
 *
 * A MessageChannel rather than setTimeout: timers in a hidden tab's worker get
 * clamped to a second, which would cost more than the generation itself.
 */
const turn = (() => {
  const channel = new MessageChannel();
  channel.port1.start();
  return () => new Promise<void>((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
})();

async function generate(id: number, params: GenerateParams): Promise<void> {
  if (!tts) throw new Error('model is not loaded');
  const { segments, voiceName, options, seed } = params;

  let voiceEmb: Float32Array | null = null;
  if (voiceName) {
    let cached = voices.get(voiceName);
    if (!cached) {
      cached = await loadVoice(base, voiceName);
      voices.set(voiceName, cached);
    }
    voiceEmb = cached;
  }

  const abort = new AbortController();
  running.set(id, abort);
  try {
    for await (const chunk of tts.synthesizeStream(
      segments, voiceEmb, options, seed, abort.signal,
    )) {
      // Transferred: the codec allocates a fresh array per chunk and never looks
      // at it again, so handing the buffer over beats copying megabytes.
      post({ type: 'chunk', id, chunk }, [chunk.buffer]);
      await turn();  // so a `cancel` sent meanwhile is actually read
      if (abort.signal.aborted) break;
    }
    post({ type: 'result', id, value: { aborted: abort.signal.aborted } });
  } finally {
    running.delete(id);
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  // Cancellation is not a request/response pair: it has to be handled while the
  // generation it targets is still in flight.
  if (message.type === 'cancel') {
    running.get(message.target)?.abort();
    return;
  }

  const { id } = message;
  try {
    switch (message.type) {
      case 'downloadInfo': {
        post({
          type: 'result', id,
          value: await downloadInfo(repoBaseUrl(message.repo || DEFAULT_REPO)),
        });
        break;
      }
      case 'clearCache': {
        await clearCache();
        post({ type: 'result', id, value: null });
        break;
      }
      case 'load': {
        const loaded = await loadModel({
          repo: message.repo || DEFAULT_REPO,
          onProgress: (progress) => post({ type: 'progress', id, progress }),
        });
        tts = loaded.tts;
        base = loaded.base;
        voices.clear();
        const info: LoadedInfo = {
          voices: loaded.voices, base: loaded.base, sampleRate: loaded.tts.sampleRate,
        };
        post({ type: 'result', id, value: info });
        break;
      }
      case 'generate': {
        await generate(id, message.params);
        break;
      }
    }
  } catch (error) {
    post({ type: 'error', id, message: (error as Error)?.message ?? String(error) });
  }
};
