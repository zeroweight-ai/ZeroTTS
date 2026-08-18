/**
 * Streaming audio playback.
 *
 * Chunks go into an AudioWorklet ring buffer rather than a series of
 * AudioBufferSourceNodes: separate nodes leave a gap or overlap at every seam,
 * audible as a click — and this model emits a chunk as often as every 80 ms.
 *
 * Two things the ring buffer has to get right, both of which corrupt audio
 * silently when wrong:
 *
 *  1. **The writer must not lap the reader.** Generation runs faster than
 *     real time (~2x), so the producer gains roughly a second of audio for every
 *     second played. A fixed buffer with an unconditional modulo write will,
 *     after buffer_length worth of lead, start overwriting samples that have not
 *     been played yet — the audio jumps or repeats mid-playback with no error
 *     anywhere. The buffer is therefore sized for the longest utterance the
 *     model can produce, and an overflow is reported rather than wrapped.
 *
 *  2. **Full must not look like empty.** With only read/write indices,
 *     `(write - read + len) % len` is 0 both when the buffer is empty and when
 *     it is exactly full, so a full buffer reads as empty and plays silence.
 *     Monotonic counters are used instead, so the two states are distinct.
 */

// max_frames (1500) / 12.5 fps = 120 s of audio is the model's ceiling; a little
// margin on top. 130 s * 48 kHz * 4 bytes ~= 25 MB, which is cheap next to the
// ~900 MB of weights already resident.
const BUFFER_SECONDS = 130;

const WORKLET_SOURCE = `
class RingBufferProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const capacity = (options && options.processorOptions && options.processorOptions.capacity)
      || 48000 * 130;
    this.buffer = new Float32Array(capacity);
    // Monotonic sample counts, NOT indices — see the module docstring. A Number
    // holds 2^53 samples exactly, which is ~5700 years at 48 kHz.
    this.written = 0;
    this.read = 0;
    this.done = false;
    this.ended = false;
    this.overflowed = false;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data.type === 'push') {
        const chunk = data.chunk;
        const free = this.buffer.length - (this.written - this.read);
        if (chunk.length > free) {
          // Dropping is bad, but silently overwriting unplayed audio is worse:
          // it corrupts the middle of the output with no signal at all.
          if (!this.overflowed) {
            this.overflowed = true;
            this.port.postMessage({ type: 'overflow' });
          }
          return;
        }
        const cap = this.buffer.length;
        let w = this.written % cap;
        for (let i = 0; i < chunk.length; i++) {
          this.buffer[w] = chunk[i];
          w = w + 1 === cap ? 0 : w + 1;
        }
        this.written += chunk.length;
      } else if (data.type === 'done') {
        this.done = true;
      } else if (data.type === 'reset') {
        this.written = 0;
        this.read = 0;
        this.done = false;
        this.ended = false;
        this.overflowed = false;
        this.buffer.fill(0);
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    const available = this.written - this.read;

    if (available === 0) {
      output.fill(0);
      if (this.done && !this.ended) {
        this.ended = true;
        this.port.postMessage({ type: 'drained' });
      }
      // Never return false: that retires the processor permanently, and the
      // node is reused for the next generation (which would then be silent).
      return true;
    }

    const cap = this.buffer.length;
    const n = Math.min(output.length, available);
    let r = this.read % cap;
    for (let i = 0; i < n; i++) {
      output[i] = this.buffer[r];
      r = r + 1 === cap ? 0 : r + 1;
    }
    this.read += n;
    for (let i = n; i < output.length; i++) output[i] = 0;
    return true;
  }
}
registerProcessor('ring-buffer', RingBufferProcessor);
`;

/** How long to wait on `resume()` before deciding the autoplay policy is not
 *  going to let it through. Long enough for a genuine resume (single-digit ms),
 *  short enough that a wedged one is not felt. */
const RESUME_TIMEOUT_MS = 1000;

/** Interactions that count as a gesture, for a second attempt at resuming. */
const GESTURES = ['pointerdown', 'keydown', 'touchend'] as const;

/** A function, not an inline `context.state === 'running'`: the state changes
 *  behind our back (that is the whole point of awaiting a resume), and reading
 *  it through a call keeps the compiler from narrowing an earlier check onto a
 *  later one. */
const running = (context: AudioContext) => context.state === 'running';

export class StreamPlayer {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  /** Whether a gesture retry is already pending, so listeners don't stack up. */
  private gestureArmed = false;

  /** Set if the producer outran the buffer — the output is then incomplete. */
  overflowed = false;

  constructor(private sampleRate: number) {}

  async start(): Promise<void> {
    this.overflowed = false;
    if (this.node) {
      this.node.port.postMessage({ type: 'reset' });
      await this.resume();
      return;
    }
    this.context = new AudioContext({ sampleRate: this.sampleRate });
    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await this.context.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    this.node = new AudioWorkletNode(this.context, 'ring-buffer', {
      outputChannelCount: [1],
      processorOptions: { capacity: Math.ceil(this.sampleRate * BUFFER_SECONDS) },
    });
    this.node.port.onmessage = (event) => {
      if (event.data?.type === 'overflow') this.overflowed = true;
    };
    this.node.connect(this.context.destination);
    await this.resume();
  }

  /**
   * Bring the context out of `suspended`, WITHOUT letting the caller hang on it.
   *
   * Autoplay policy: a context created outside a user gesture starts suspended,
   * and then nothing is ever heard — hence the resume. The trap is what happens
   * when the policy is not satisfied: `resume()` returns a promise that is not
   * rejected and not resolved, it simply stays pending until a gesture arrives,
   * possibly forever. Awaited directly, that pending promise wedges the whole
   * generation before the first frame — the run never starts, never fails, and
   * the button stays disabled with no way out but a reload.
   *
   * Two paths reach it in practice: a click synthesized by script (no gesture at
   * all), and iOS Safari, where the gesture is consumed by the `await` on
   * `addModule` above and is no longer "recent" by the time resume is called.
   *
   * So: give it a moment, then continue regardless. Generation proceeds, the WAV
   * comes out complete either way, and a listener picks the audio back up on the
   * next thing the user touches.
   */
  private async resume(): Promise<void> {
    const context = this.context;
    if (!context || running(context)) return;

    let timer = 0;
    await Promise.race([
      context.resume(),
      new Promise<void>((r) => { timer = self.setTimeout(r, RESUME_TIMEOUT_MS); }),
    ]);
    clearTimeout(timer);

    if (!running(context)) this.resumeOnGesture(context);
  }

  /** Retry the resume on the user's next interaction, once. */
  private resumeOnGesture(context: AudioContext): void {
    if (this.gestureArmed) return;
    this.gestureArmed = true;
    const retry = () => {
      this.gestureArmed = false;
      for (const event of GESTURES) removeEventListener(event, retry);
      // Inside the gesture's own task, which is the whole point.
      context.resume().catch(() => { /* a closed context is not worth a report */ });
    };
    for (const event of GESTURES) addEventListener(event, retry, { once: true });
  }

  push(chunk: Float32Array): void {
    // Not transferred: the caller keeps every chunk to assemble the WAV
    // download, and transferring would detach the buffer out from under it.
    this.node?.port.postMessage({ type: 'push', chunk });
  }

  /** No more chunks are coming; playback ends when the buffer drains. */
  finish(): void {
    this.node?.port.postMessage({ type: 'done' });
  }

  async stop(): Promise<void> {
    this.node?.port.postMessage({ type: 'reset' });
    await this.context?.suspend();
  }
}

/** Wrap mono float32 samples as a WAV blob, for the download button. */
export function toWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
