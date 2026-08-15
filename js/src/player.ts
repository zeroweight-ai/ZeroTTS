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

export class StreamPlayer {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;

  /** Set if the producer outran the buffer — the output is then incomplete. */
  overflowed = false;

  constructor(private sampleRate: number) {}

  async start(): Promise<void> {
    this.overflowed = false;
    if (this.node) {
      this.node.port.postMessage({ type: 'reset' });
      await this.context?.resume();
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
    // Autoplay policy: a context created outside a user gesture starts
    // suspended, and then nothing is ever heard.
    await this.context.resume();
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
