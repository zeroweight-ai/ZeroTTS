/**
 * Punctuation normalization + sentence chunking — port of zerotts/chunking.py.
 *
 * The model is trained on utterances, not paragraphs, and generation is bounded
 * by `maxFrames` (1500 frames = 120 s). Feeding it a whole article as one
 * utterance does not error — it just stops mid-sentence when the frame budget
 * runs out. Long text has to be split.
 *
 * Splitting falls back through a hierarchy so no segment exceeds the budget,
 * however the input is (or is not) punctuated:
 *   1. sentence-ending punctuation (. ! ? …)
 *   2. commas, if a "sentence" is still too long
 *   3. word boundaries, if a comma-separated piece is still too long
 *   4. raw character slicing, for a single unbroken run
 *
 * Keep this in step with the Python version: the two are expected to segment
 * identically, and a divergence shows up as the browser and the package
 * producing different audio for the same input.
 */

const SENTENCE_END = /(?<=[.!?…])\s+/;
const COMMA = /(?<=,)\s+/;

/** Rough speaking rate, used to size segments by estimated duration. */
const CHARS_PER_SEC = 15.0;

const EXISTING_PUNCT = new Set([...'.!?…,;:—–-"\')]}']);

/**
 * Rewrite written-only punctuation into breaks the model can voice.
 *
 *   ';' -> ','       the vocab has the token, but the training corpus is
 *                    transcribed speech where it barely occurs, so the model has
 *                    no reliable prosody for it.
 *   newline -> '. '  unless the line already ends in punctuation. The tokenizer
 *                    collapses all whitespace to a single space, so an
 *                    un-rewritten line break vanishes entirely and two unrelated
 *                    lines run together in one breath.
 *
 * Semicolons are converted first, so a line ending in one already counts as
 * punctuated when the newline rule runs and does not also collect a '.'.
 */
export function normalizePunctuation(text: string): string {
  if (!text) return text;
  let out = text.replace(/;/g, ',');
  out = out.replace(/(.?)[ \t]*(?:\r?\n[ \t]*)+/g, (_m, prev: string) => {
    if (!prev) return '';                       // leading newline: nothing to punctuate
    if (EXISTING_PUNCT.has(prev)) return `${prev} `;
    return `${prev}. `;
  });
  return out.trim();
}

function splitSentences(text: string): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const sentences: string[] = [];
  for (const para of trimmed.split(/\n\s*\n/)) {
    const p = para.trim();
    if (!p) continue;
    for (const sent of p.split(SENTENCE_END)) {
      const s = sent.trim();
      if (s) sentences.push(s);
    }
  }
  return sentences;
}

/** Greedily join pieces (each already <= maxChars) as close to maxChars as possible. */
function packPieces(pieces: string[], maxChars: number): string[] {
  const chunks: string[] = [];
  let current: string[] = [];
  let currentLen = 0;
  for (const piece of pieces) {
    let pieceLen = piece.length + (current.length ? 1 : 0);
    if (current.length && currentLen + pieceLen > maxChars) {
      chunks.push(current.join(' '));
      current = [];
      currentLen = 0;
      pieceLen = piece.length;
    }
    current.push(piece);
    currentLen += pieceLen;
  }
  if (current.length) chunks.push(current.join(' '));
  return chunks;
}

function splitByChars(text: string, maxChars: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += maxChars) out.push(text.slice(i, i + maxChars));
  return out;
}

function splitByWords(text: string, maxChars: number): string[] {
  const packed = packPieces(text.split(/\s+/).filter(Boolean), maxChars);
  const out: string[] = [];
  for (const piece of packed) {
    if (piece.length <= maxChars) out.push(piece);
    else out.push(...splitByChars(piece, maxChars));
  }
  return out;
}

/** Break text down until every piece is <= maxChars, least disruptive split first. */
function atomize(text: string, maxChars: number): string[] {
  const t = text.trim();
  if (!t) return [];
  if (t.length <= maxChars) return [t];

  const commaParts = t.split(COMMA).map((p) => p.trim()).filter(Boolean);
  if (commaParts.length > 1) {
    const out: string[] = [];
    for (const part of commaParts) out.push(...atomize(part, maxChars));
    return out;
  }
  if (/\s/.test(t)) return splitByWords(t, maxChars);
  return splitByChars(t, maxChars);
}

export function chunkText(text: string, maxChunkSec = 15.0): string[] {
  const maxChars = Math.max(1, Math.floor(maxChunkSec * CHARS_PER_SEC));
  const atoms: string[] = [];
  for (const sentence of splitSentences(text)) atoms.push(...atomize(sentence, maxChars));
  return packPieces(atoms, maxChars);
}

// ── per-segment punctuation cleanup ─────────────────────────────────────────
// Chunking can cut a sentence mid-flow, leaving a segment that ends on a comma
// or nothing, and/or a stray terminal mark stranded in the middle of what is now
// a standalone utterance. Interior punctuation reads as a hard stop where only a
// pause belongs, so mid-segment marks are downgraded to a comma-pause and the
// segment's own end gets a real terminator.

const TRAILING = /[^\p{L}\p{N}_]+$/u;
const MID_PUNCT = /[^\p{L}\p{N}_\s\-/.,:?@!"'%]/gu;
const REPEAT_COMMA = /\s*(?:,\s*)+/g;
const END_PUNCT = new Set(['.', '!', '?']);

export function cleanSegmentPunctuation(text: string): string {
  const t = text.trim();
  if (!t) return t;

  const match = t.match(TRAILING);
  const core0 = match ? t.slice(0, match.index) : t;
  const trailing = match ? (match[0] ?? '').trimEnd() : '';

  const endPunct = trailing && END_PUNCT.has(trailing[trailing.length - 1])
    ? trailing[trailing.length - 1]
    : '.';

  let core = core0.replace(MID_PUNCT, ',');
  core = core.replace(REPEAT_COMMA, ', ').replace(/^[\s,]+|[\s,]+$/g, '');
  if (!core) return '';
  return core + endPunct;
}

/** The exact segments the synthesizer will speak, chunked and cleaned. */
export function textSegments(text: string, maxChunkSec = 15.0): string[] {
  return chunkText(normalizePunctuation(text), maxChunkSec)
    .map(cleanSegmentPunctuation)
    .filter(Boolean);
}

/** Parse a "### name"-delimited sample-text file into {name: text}. */
export function parseSamples(raw: string): Record<string, string> {
  const samples: Record<string, string> = {};
  let name: string | null = null;
  let lines: string[] = [];
  const flush = () => {
    if (name !== null) samples[name] = lines.join('\n').trim();
  };
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith('### ')) {
      flush();
      name = line.slice(4).trim();
      lines = [];
    } else if (line.startsWith('#')) {
      continue;
    } else {
      lines.push(line);
    }
  }
  flush();
  return samples;
}
