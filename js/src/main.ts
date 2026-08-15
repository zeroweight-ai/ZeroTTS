/** Demo UI wiring. The interesting code is in synthesizer.ts / codec.ts. */

import { clearCache } from './cache';
import { downloadInfo, loadModel, loadVoice, repoBaseUrl, DEFAULT_REPO } from './loader';
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
  status: $<HTMLDivElement>('status'),
  bar: $<HTMLDivElement>('bar'),
  barWrap: $<HTMLDivElement>('bar-wrap'),
  sizeNote: $<HTMLDivElement>('size-note'),
};

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

  try {
    status('Loading voice…');
    const voiceEmb = voiceName ? await loadVoice(base, voiceName) : null;

    await player.start();
    status('Generating…');

    for await (const chunk of tts.synthesizeStream(
      text, voiceEmb,
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
    status(`${duration.toFixed(2)}s audio in ${elapsed.toFixed(2)}s — ` +
           `${(duration / elapsed).toFixed(1)}x realtime, ` +
           `first audio ${firstChunkAt?.toFixed(0) ?? '?'} ms`);

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

els.repo.addEventListener('change', refreshSizeNote);
refreshSizeNote();
