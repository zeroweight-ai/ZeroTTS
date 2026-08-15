/**
 * Streaming audio playback.
 *
 * Chunks are pushed into an AudioWorklet ring buffer rather than scheduled as
 * individual AudioBufferSourceNodes: separate nodes leave gaps or overlaps at
 * chunk seams, which is audible as a click on every boundary — and this model
 * emits a chunk as often as every 80 ms.
 */

const WORKLET_SOURCE = `
class RingBufferProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // ~10 s at 48 kHz. Generation is faster than realtime, so the buffer has to
    // absorb the lead rather than drop it.
    this.buffer = new Float32Array(480000);
    this.readIndex = 0;
    this.writeIndex = 0;
    this.done = false;
    this.port.onmessage = (event) => {
      if (event.data.type === 'push') {
        const chunk = event.data.chunk;
        for (let i = 0; i < chunk.length; i++) {
          this.buffer[this.writeIndex] = chunk[i];
          this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        }
      } else if (event.data.type === 'done') {
        this.done = true;
      } else if (event.data.type === 'reset') {
        this.readIndex = 0; this.writeIndex = 0; this.done = false;
        this.buffer.fill(0);
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    const available = (this.writeIndex - this.readIndex + this.buffer.length) % this.buffer.length;
    if (available === 0) {
      output.fill(0);
      if (this.done) { this.port.postMessage({ type: 'drained' }); return false; }
      return true;
    }
    const n = Math.min(output.length, available);
    for (let i = 0; i < n; i++) {
      output[i] = this.buffer[this.readIndex];
      this.readIndex = (this.readIndex + 1) % this.buffer.length;
    }
    for (let i = n; i < output.length; i++) output[i] = 0;
    return true;
  }
}
registerProcessor('ring-buffer', RingBufferProcessor);
`;

export class StreamPlayer {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;

  constructor(private sampleRate: number) {}

  async start(): Promise<void> {
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
    this.node = new AudioWorkletNode(this.context, 'ring-buffer', { outputChannelCount: [1] });
    this.node.connect(this.context.destination);
  }

  push(chunk: Float32Array): void {
    this.node?.port.postMessage({ type: 'push', chunk }, []);
  }

  /** Signal that no more chunks are coming; playback ends when the buffer drains. */
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
