/** Demo UI wiring. The interesting code is in synthesizer.ts / codec.ts.
 *
 *  The page mirrors webui/app.py: a basic view that is model → voice → text →
 *  player and nothing else, with the run log, the segments and every sampling
 *  knob folded into "Tuỳ chọn nâng cao". Templates and the take history are
 *  lists of real rows with a preview, not a strip of tags.
 *
 *  Nothing here imports onnxruntime-web, and that is load-bearing: the model
 *  runs in a worker (worker.ts) so that a generation — minutes of blocking WASM
 *  compute — never touches this thread. This file only ever handles text going
 *  out and finished audio chunks coming back.
 */

import bannerUrl from '../../docs/assets/banner.png';

import { fetchWithCache } from './cache';
import { textSegments } from './chunking';
import { normalizeViText } from './textNorm';
import { DEFAULT_REPO, voicePreviewUrl } from './repo';
import { loadSampleTexts } from './samples';
import { StreamPlayer, toWavBlob } from './player';
import { TtsWorker } from './workerClient';
import { VoiceIndex } from './types';

/** Same first-load text as the Gradio UI. */
const DEFAULT_TEXT = 'Xin chào tất cả mọi người. Giọng nói này được tạo ra bởi ZeroTTS.';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  banner: $<HTMLImageElement>('banner'),
  load: $<HTMLButtonElement>('load'),
  generate: $<HTMLButtonElement>('generate'),
  stop: $<HTMLButtonElement>('stop'),
  download: $<HTMLAnchorElement>('download'),
  clear: $<HTMLButtonElement>('clear-cache'),
  text: $<HTMLTextAreaElement>('text'),
  voice: $<HTMLSelectElement>('voice'),
  repo: $<HTMLInputElement>('repo'),
  seed: $<HTMLInputElement>('seed'),
  cfg: $<HTMLInputElement>('cfg'),
  temperature: $<HTMLInputElement>('temperature'),
  chunkSec: $<HTMLInputElement>('chunk-sec'),
  textNorm: $<HTMLInputElement>('text-norm'),
  status: $<HTMLDivElement>('status'),
  bar: $<HTMLDivElement>('bar'),
  barWrap: $<HTMLDivElement>('bar-wrap'),
  sizeNote: $<HTMLParagraphElement>('size-note'),
  preview: $<HTMLAudioElement>('preview'),
  voiceMeta: $<HTMLParagraphElement>('voice-meta'),
  segments: $<HTMLPreElement>('segments'),
  result: $<HTMLAudioElement>('result'),
  livePill: $<HTMLDivElement>('live-pill'),
  history: $<HTMLDivElement>('history'),
  templates: $<HTMLDivElement>('templates'),
};

interface Take { url: string; title: string; text: string; }

let samples: Record<string, string> = {};

const tts = new TtsWorker();
let sampleRate = 0;
let voices: VoiceIndex = { voices: [] };
let base = '';
let player: StreamPlayer | null = null;
let cancelRun: (() => void) | null = null;
/** Set by Stop, so the run that unwinds afterwards knows not to report itself. */
let stopped = false;
const takes: Take[] = [];

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`;

function status(text: string): void {
  els.status.textContent = text;
}

function progress(fraction: number | null): void {
  els.barWrap.style.display = fraction === null ? 'none' : 'block';
  if (fraction !== null) els.bar.style.width = `${Math.round(fraction * 100)}%`;
}

function live(on: boolean): void {
  els.livePill.classList.toggle('on', on);
}

async function refreshSizeNote(): Promise<void> {
  try {
    const info = await tts.downloadInfo(els.repo.value || DEFAULT_REPO);
    els.sizeNote.textContent = info.cached
      ? `Mô hình đã có sẵn trên máy (${mb(info.bytes)}) — tải sẽ rất nhanh.`
      : `Lần đầu sẽ tải khoảng ${mb(info.bytes)} (fp32, chưa lượng tử hoá) và ` +
        `lưu lại cho những lần sau. Nên dùng máy tính với mạng nhanh.`;
  } catch {
    els.sizeNote.textContent =
      'Lần đầu sẽ tải khoảng 900 MB (fp32, chưa lượng tử hoá) và lưu lại cho những lần sau.';
  }
}

els.load.addEventListener('click', async () => {
  els.load.disabled = true;
  try {
    status('Đang tải mô hình…');
    progress(0);
    const loaded = await tts.load(els.repo.value || DEFAULT_REPO, (p) => {
      if (p.overallTotal > 0) progress(p.overallLoaded / p.overallTotal);
      els.sizeNote.textContent =
        `Đang tải ${p.file.split('/').pop()} — ${mb(p.overallLoaded)} / ${mb(p.overallTotal)}`;
    });
    voices = loaded.voices;
    base = loaded.base;
    sampleRate = loaded.sampleRate;

    renderVoiceOptions();
    // Prefer "maichi" (Mai Chi) as the default, same as the README/webui — its
    // dropdown position is not guaranteed to be first once a repo ships more
    // presets than the shipped index happens to list it first.
    if (voices.voices.some((v) => v.name === 'maichi')) els.voice.value = 'maichi';
    else if (voices.voices.length) els.voice.value = voices.voices[0].name;
    updateVoiceUi();

    player = new StreamPlayer(sampleRate);
    progress(null);
    els.sizeNote.textContent =
      `Đã sẵn sàng — ${voices.voices.length} giọng, ${sampleRate / 1000} kHz.`;
    status(`Ready — ${voices.voices.length} voice(s), ${sampleRate / 1000} kHz.`);
    els.voice.disabled = false;
    els.generate.disabled = false;
    els.load.textContent = '✓  Đã tải mô hình';
  } catch (error) {
    progress(null);
    els.sizeNote.textContent = `Tải mô hình thất bại: ${(error as Error).message}`;
    status(`Load failed: ${(error as Error).message}`);
    els.load.disabled = false;
  }
});

els.generate.addEventListener('click', async () => {
  if (!player) return;
  const text = els.text.value.trim();
  if (!text) { status('Hãy nhập văn bản trước.'); return; }

  els.generate.disabled = true;
  els.stop.disabled = false;
  els.download.style.display = 'none';
  stopped = false;

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
    await player.start();
    status('Generating…');
    live(true);

    // The worker loads the voice and runs the model; this thread stays free to
    // paint, so the buttons and the log update while generation is under way.
    const run = tts.generate({
      segments, voiceName,
      options: {
        cfgScale: Number(els.cfg.value), audioTemperature: Number(els.temperature.value),
      },
      seed,
    });
    cancelRun = run.cancel;

    for await (const chunk of run.chunks) {
      if (firstChunkAt === null) {
        firstChunkAt = performance.now() - started;
        status(`Playing — first audio in ${firstChunkAt.toFixed(0)} ms`);
      }
      chunks.push(chunk);
      player.push(chunk);
    }
    player.finish();
    // Stop already reset the player and said so; a partial take is not worth
    // overwriting that with statistics.
    if (stopped) return;

    const total = chunks.reduce((n, c) => n + c.length, 0);
    const audio = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) { audio.set(c, offset); offset += c.length; }

    const elapsed = (performance.now() - started) / 1000;
    const duration = total / sampleRate;
    const overflow = player.overflowed
      ? ' — WARNING: playback buffer overflowed, live audio is incomplete (the ' +
        'downloaded WAV is not)'
      : '';
    status(`${duration.toFixed(2)}s audio in ${elapsed.toFixed(2)}s — ` +
           `${(duration / elapsed).toFixed(1)}x realtime, ` +
           `first audio ${firstChunkAt?.toFixed(0) ?? '?'} ms, ` +
           `${segments.length} segment(s)${overflow}`);

    // One blob per take, kept for the session: it is both the main player's
    // source and the history entry's, so it must outlive this handler.
    const url = URL.createObjectURL(toWavBlob(audio, sampleRate));
    els.download.href = url;
    els.download.download = 'zerotts.wav';
    els.download.style.display = 'inline-block';
    showTake(url, false);
    addTake({ url, text, title: takeTitle(voiceName, duration) });
  } catch (error) {
    if (!stopped) status(`Generation failed: ${(error as Error).message}`);
  } finally {
    live(false);
    els.generate.disabled = false;
    els.stop.disabled = true;
    cancelRun = null;
  }
});

els.stop.addEventListener('click', async () => {
  stopped = true;
  cancelRun?.();
  await player?.stop();
  live(false);
  status('Stopped.');
});

els.clear.addEventListener('click', async () => {
  await tts.clearCache();
  await refreshSizeNote();
  status('Cache cleared. The next load will re-download the model.');
});

/** Populate the voice <select> from `voices.voices`. No tag filter: with ~9
 *  shipped packs the labels already carry the tags, and a filter above the
 *  picker is one more thing to understand before hearing anything. */
function renderVoiceOptions(): void {
  els.voice.innerHTML = '<option value="">(không dùng giọng nào)</option>';
  for (const v of voices.voices) {
    const option = document.createElement('option');
    option.value = v.name;
    const label = v.display_name || v.name;
    option.textContent = (v.tags && v.tags.length) ? `${label} — ${v.tags.join(', ')}` : label;
    els.voice.append(option);
  }
}

/** Preview clips already fetched, as object URLs. */
const previewUrls = new Map<string, string>();
/** Bumped on every voice change so a slow fetch cannot land on a newer voice. */
let previewToken = 0;

async function updateVoiceUi(): Promise<void> {
  const name = els.voice.value;
  const token = ++previewToken;
  if (!name || !base) {
    els.preview.removeAttribute('src');
    els.preview.style.display = 'none';
    els.voiceMeta.textContent = name ? '' : 'Không chọn giọng — mô hình tự chọn một '
      + 'giọng, và giọng đó không giống nhau giữa các đoạn.';
    return;
  }
  const info = voices.voices.find((v) => v.name === name);
  els.voiceMeta.textContent = info
    ? [info.display_name || info.name, info.language, ...(info.tags ?? [])]
        .filter(Boolean).join(' · ')
    : name;

  // Fetch the clip whole and hand the element a blob, rather than pointing it
  // at the CDN and letting it stream: a progressive fetch that stalls or gets
  // cancelled — which is easy while the model download is saturating the
  // connection — fires `error` mid-playback, and the audio would cut out and
  // vanish. A blob is either there or it never appears.
  els.preview.style.display = 'block';
  try {
    let url = previewUrls.get(name);
    if (!url) {
      const buf = await fetchWithCache(voicePreviewUrl(base, name));
      url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
      previewUrls.set(name, url);
    }
    if (token !== previewToken) return;  // the user moved on while we fetched
    els.preview.src = url;
  } catch {
    // No preview shipped for this voice — an empty player, not a broken one.
    if (token === previewToken) els.preview.style.display = 'none';
  }
}

// ── the two list panels ──────────────────────────────────────────────────────

/** One clickable row: a bold title over a single-line preview. */
function listItem(title: string, preview: string, onClick: () => void): HTMLButtonElement {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'item';
  const t = document.createElement('div');
  t.className = 't';
  t.textContent = title;
  const p = document.createElement('div');
  p.className = 'p';
  p.textContent = preview;
  item.append(t, p);
  item.addEventListener('click', onClick);
  return item;
}

/** Collapse a sample down to the one line a row shows (the CSS ellipsises it). */
const oneLine = (text: string) => text.replace(/\s+/g, ' ').trim();

function renderTemplates(): void {
  els.templates.replaceChildren(
    ...Object.entries(samples).map(([name, text]) =>
      listItem(name, oneLine(text), () => {
        els.text.value = text;
        els.text.scrollIntoView({ behavior: 'smooth', block: 'center' });
      })),
  );
}

function takeTitle(voiceName: string, seconds: number): string {
  const voice = voices.voices.find((v) => v.name === voiceName);
  const label = voice ? (voice.display_name || voice.name) : 'không giọng';
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  return `🔊  ${label}  ·  ${now}  ·  ${seconds.toFixed(0)}s`;
}

/**
 * Load a take into the main player — the one place audio comes from on this
 * page, so a history row never spawns a second player of its own.
 *
 * `autoplay` is false for a take that was just generated. Generation runs
 * faster than real time, so when the last chunk is produced the live ring
 * buffer still has seconds of audio left to play; starting the finished WAV
 * from zero at that moment plays the take on top of itself.
 */
function showTake(url: string, autoplay: boolean): void {
  els.result.src = url;
  if (autoplay) {
    els.result.play().catch(() => { /* autoplay may be blocked; controls still work */ });
  }
}

function addTake(take: Take): void {
  takes.unshift(take);
  renderHistory();
}

function renderHistory(): void {
  if (!takes.length) {
    els.history.innerHTML = '<div class="empty">Chưa có bản nào trong phiên này.</div>';
    return;
  }
  els.history.replaceChildren(
    ...takes.map((t) => listItem(t.title, oneLine(t.text), () => showTake(t.url, true))),
  );
}

els.voice.addEventListener('change', updateVoiceUi);
els.repo.addEventListener('change', refreshSizeNote);

els.banner.src = bannerUrl;
els.text.value = DEFAULT_TEXT;
refreshSizeNote();
loadSampleTexts().then((loaded) => {
  samples = loaded;
  renderTemplates();
});
