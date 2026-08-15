/**
 * Fetching model files with progress, and persisting them across visits.
 *
 * The weights are fp32 and unquantized — roughly 900 MB total. That is a
 * deliberate quality-over-size choice (see docs/BROWSER.md), and it makes
 * persistence mandatory rather than a nicety: nobody will download this twice.
 *
 * Cache API is used rather than OPFS because it stores Response objects
 * directly, survives reloads, and needs no manual quota bookkeeping.
 */

const CACHE_NAME = 'zerotts-weights-v1';

export interface DownloadProgress {
  file: string;
  loaded: number;
  total: number;
  /** Bytes across all files in the current batch, for an aggregate bar. */
  overallLoaded: number;
  overallTotal: number;
}

export type ProgressFn = (p: DownloadProgress) => void;

async function openCache(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    // Private browsing or a blocked storage policy — still works, just slow
    // on every visit.
    return null;
  }
}

/** Fetch a URL as an ArrayBuffer, reporting progress and caching the result. */
export async function fetchWithCache(
  url: string,
  onProgress?: ProgressFn,
  overall = { loaded: 0, total: 0 },
): Promise<ArrayBuffer> {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    overall.loaded += buf.byteLength;
    onProgress?.({
      file: url, loaded: buf.byteLength, total: buf.byteLength,
      overallLoaded: overall.loaded, overallTotal: overall.total,
    });
    return buf;
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);

  const total = responseSize(response.headers);
  const reader = response.body?.getReader();

  // No streaming body (or no reader): fall back to a single read, losing
  // per-file progress but not correctness.
  if (!reader) {
    const buf = await response.arrayBuffer();
    await cache?.put(url, new Response(buf));
    return buf;
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.({
      file: url, loaded, total,
      overallLoaded: overall.loaded + loaded, overallTotal: overall.total,
    });
  }
  overall.loaded += loaded;

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  await cache?.put(url, new Response(out.slice(0)));
  return out.buffer;
}

/**
 * Size of a response, preferring Hugging Face's `x-linked-size`.
 *
 * Big files on the Hub are Git LFS pointers served through a CDN redirect. On
 * those, `content-length` can describe the pointer/redirect rather than the real
 * payload, so reading it alone reports a few hundred bytes for a 391 MB graph
 * and the progress bar is nonsense. `x-linked-size` is the actual object size.
 */
function responseSize(headers: Headers): number {
  return Number(headers.get('x-linked-size') ?? headers.get('content-length') ?? 0);
}

/** HEAD every URL to size the aggregate progress bar before downloading. */
export async function totalBytes(urls: string[]): Promise<number> {
  let total = 0;
  await Promise.all(urls.map(async (url) => {
    try {
      const r = await fetch(url, { method: 'HEAD' });
      total += responseSize(r.headers);
    } catch {
      /* an unknown size just makes the bar less precise */
    }
  }));
  return total;
}

/** Whether every URL is already cached — lets the UI skip the size warning. */
export async function isCached(urls: string[]): Promise<boolean> {
  const cache = await openCache();
  if (!cache) return false;
  const hits = await Promise.all(urls.map((u) => cache.match(u)));
  return hits.every(Boolean);
}

export async function clearCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}
