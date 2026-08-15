/** Demo UI wiring. The interesting code is in synthesizer.ts / codec.ts. */

import { clearCache } from './cache';
import { textSegments } from './chunking';
import { normalizeViText } from './textNorm';
import {
  DEFAULT_REPO, downloadInfo, loadModel, loadSampleTexts, loadVoice,
  repoBaseUrl, voicePreviewUrl,
} from './loader';
import { StreamPlayer, toWavBlob } from './player';
import { ZeroTTSBrowser } from './synthesizer';
import { VoiceIndex } from './types';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  load: $<HTMLButtonElement>('load'),
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  download: $<HTMLAnchorElement>('download'),
  clear: $<HTMLButtonElement>('clear-cache'),
  text: $<HTMLTextAreaElement>('text'),
  voice: $<HTMLSelectElement>('voice'),
  repo: $<HTMLInputElement>('repo'),
  ep: $<HTMLSelectElement>('ep'),
  seed: $<HTMLInputElement>('seed'),
  cfg: $<HTMLInputElement>('cfg'),
  temperature: $<HTMLInputElement>('temperature'),
  chunkSec: $<HTMLInputElement>('chunk-sec'),
  textNorm: $<HTMLInputElement>('text-norm'),
  status: $<HTMLDivElement>('status'),
  bar: $<HTMLDivElement>('bar'),
  barWrap: $<HTMLDivElement>('bar-wrap'),
  sizeNote: $<HTMLDivElement>('size-note'),
  sample: $<HTMLSelectElement>('sample'),
  preview: $<HTMLAudioElement>('preview'),
  voiceMeta: $<HTMLDivElement>('voice-meta'),
  segments: $<HTMLPreElement>('segments'),
};

let samples: Record<string, string> = {};

let tts: ZeroTTSBrowser | null = null;
let voices: VoiceIndex = { voices: [] };
let base = '';
let player: StreamPlayer | null = null;
let abort: AbortController | null = null;

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;

function status(text: string): void {
  els.status.textContent = text;
}

function progress(fraction: number | null): void {
  els.barWrap.style.display = fraction === null ? 'none' : 'block';
  if (fraction !== null) els.bar.style.width = `${Math.round(fraction * 100)}%`;
}

async function refreshSizeNote(): Promise<void> {
  try {
    const info = await downloadInfo(repoBaseUrl(els.repo.value || DEFAULT_REPO));
    els.sizeNote.textContent = info.cached
      ? `Model is cached locally (${mb(info.bytes)}) — loading will be fast.`
      : `First load downloads ~${mb(info.bytes)} (fp32, not quantized). ` +
        `It is cached afterwards. Desktop broadband recommended.`;
  } catch {
    els.sizeNote.textContent =
      'First load downloads ~900 MB (fp32, not quantized) and caches it afterwards.';
  }
}

els.load.addEventListener('click', async () => {
  els.load.disabled = true;
  try {
    status('Downloading model…');
    progress(0);
    const loaded = await loadModel({
      repo: els.repo.value || DEFAULT_REPO,
      executionProvider: els.ep.value as 'wasm' | 'webgpu',
      onProgress: (p) => {
        if (p.overallTotal > 0) progress(p.overallLoaded / p.overallTotal);
        status(`Downloading ${p.file.split('/').pop()} — ` +
               `${mb(p.overallLoaded)} / ${mb(p.overallTotal)}`);
      },
    });
    tts = loaded.tts;
    voices = loaded.voices;
    base = loaded.base;

    els.voice.innerHTML = '<option value="">(unconditional voice)</option>';
    for (const v of voices.voices) {
      const option = document.createElement('option');
      option.value = v.name;
      option.textContent = v.description ? `${v.name} — ${v.description}` : v.name;
      els.voice.append(option);
    }
    if (voices.voices.length) els.voice.value = voices.voices[0].name;
    updateVoiceUi();

    player = new StreamPlayer(tts.sampleRate);
    progress(null);
    status(`Ready — ${voices.voices.length} voice(s), ${tts.sampleRate / 1000} kHz.`);
    els.generate.disabled = false;
  } catch (error) {
    progress(null);
    status(`Load failed: ${(error as Error).message}`);
    els.load.disabled = false;
  }
});

els.generate.addEventListener('click', async () => {
  if (!tts || !player) return;
  const text = els.text.value.trim();
  if (!text) { status('Enter some text first.'); return; }

  els.generate.disabled = true;
  els.stop.disabled = false;
  els.download.style.display = 'none';
  abort = new AbortController();

  const seedValue = Number(els.seed.value);
  const seed = Number.isFinite(seedValue) && seedValue >= 0 ? seedValue : undefined;

  const voiceName = els.voice.value;
  const chunks: Float32Array[] = [];
  const started = performance.now();
  let firstChunkAt: number | null = null;

  // Normalize BEFORE segmenting — an expansion is several times longer than
  // what it replaces, and the chunk budget has to size the text the model
  // actually receives. Same order as the Python engine.
  const normalized = els.textNorm.checked ? normalizeViText(text) : text;
  // The model is bounded by maxFrames (120 s); a whole article as one utterance
  // is silently truncated mid-sentence.
  const segments = textSegments(normalized, Number(els.chunkSec.value));
  els.segments.textContent = segments.map((s, i) => `[${i + 1}] ${s}`).join('\n');

  try {
    status('Loading voice…');
    const voiceEmb = voiceName ? await loadVoice(base, voiceName) : null;

    await player.start();
    status('Generating…');

    for await (const chunk of tts.synthesizeStream(
      segments, voiceEmb,
      { cfgScale: Number(els.cfg.value), audioTemperature: Number(els.temperature.value) },
      seed, abort.signal,
    )) {
      if (firstChunkAt === null) {
        firstChunkAt = performance.now() - started;
        status(`Playing — first audio in ${firstChunkAt.toFixed(0)} ms`);
      }
      chunks.push(chunk);
      player.push(chunk);
    }
    player.finish();

    const total = chunks.reduce((n, c) => n + c.length, 0);
    const audio = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) { audio.set(c, offset); offset += c.length; }

    const elapsed = (performance.now() - started) / 1000;
    const duration = total / tts.sampleRate;
    const overflow = player.overflowed
      ? ' — WARNING: playback buffer overflowed, live audio is incomplete (the ' +
        'downloaded WAV is not)'
      : '';
    status(`${duration.toFixed(2)}s audio in ${elapsed.toFixed(2)}s — ` +
           `${(duration / elapsed).toFixed(1)}x realtime, ` +
           `first audio ${firstChunkAt?.toFixed(0) ?? '?'} ms, ` +
           `${segments.length} segment(s)${overflow}`);

    if (els.download.href) URL.revokeObjectURL(els.download.href);
    els.download.href = URL.createObjectURL(toWavBlob(audio, tts.sampleRate));
    els.download.download = 'zerotts.wav';
    els.download.style.display = 'inline-block';
  } catch (error) {
    if (!abort.signal.aborted) status(`Generation failed: ${(error as Error).message}`);
  } finally {
    els.generate.disabled = false;
    els.stop.disabled = true;
    abort = null;
  }
});

els.stop.addEventListener('click', async () => {
  abort?.abort();
  await player?.stop();
  status('Stopped.');
});

els.clear.addEventListener('click', async () => {
  await clearCache();
  await refreshSizeNote();
  status('Cache cleared. The next load will re-download the model.');
});

function updateVoiceUi(): void {
  const name = els.voice.value;
  if (!name || !base) {
    els.preview.removeAttribute('src');
    els.preview.style.display = 'none';
    els.voiceMeta.textContent = name ? '' : 'No voice — the model picks one itself, '
      + 'and it is not consistent between segments.';
    return;
  }
  const info = voices.voices.find((v) => v.name === name);
  els.voiceMeta.textContent = info
    ? [info.name, info.language, info.description].filter(Boolean).join(' · ')
    : name;
  // A 404 here should leave an empty player, not a broken one.
  els.preview.onerror = () => { els.preview.style.display = 'none'; };
  els.preview.src = voicePreviewUrl(base, name);
  els.preview.style.display = 'block';
}

els.voice.addEventListener('change', updateVoiceUi);
els.sample.addEventListener('change', () => {
  const text = samples[els.sample.value];
  if (text) els.text.value = text;
});
els.repo.addEventListener('change', refreshSizeNote);

refreshSizeNote();
loadSampleTexts().then((loaded) => {
  samples = loaded;
  for (const name of Object.keys(loaded)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    els.sample.append(option);
  }
});
