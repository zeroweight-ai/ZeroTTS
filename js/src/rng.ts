/**
 * Seedable PRNG.
 *
 * Sampling happens inside `local_frame_decode.onnx`, which takes its random
 * draws as graph inputs (`ctrl_random_u`, `audio_random_u`). That means the
 * caller owns the randomness — so a seeded generator here makes browser output
 * reproducible AND makes this port bit-comparable against the Python runtime:
 * same text, same voice, same draw sequence must give identical frame codes.
 * That equivalence is the only practical way to keep the port honest across
 * re-exports.
 *
 * mulberry32: small, fast, good enough for sampling. Not cryptographic.
 */
export class Rng {
  private state: number;

  constructor(seed?: number) {
    this.state = (seed ?? Math.floor(Math.random() * 0xffffffff)) >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  fill(out: Float32Array): Float32Array {
    for (let i = 0; i < out.length; i++) out[i] = this.next();
    return out;
  }
}
