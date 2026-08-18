/**
 * Fetching model files with progress, and persisting them across visits.
 *
 * The weights are fp32 and unquantized — roughly 900 MB total. That is a
 * deliberate quality-over-size choice (see docs/BROWSER.md), and it makes
 * persistence mandatory rather than a nicety: nobody will download this twice.
 *
 * Cache API is used rather than OPFS because it stores Response objects
 * directly, survives reloads, and needs no manual quota bookkeeping.
 *
 * **Entries are keyed by ETag, not by URL.** Every file is fetched from
 * `.../resolve/main/...`, which is a MOVING target: publish a new revision on
 * the Hub and the URL is unchanged while its contents are not. Keyed by URL
 * alone, a cached copy would win forever — a re-fit voice would never reach
 * anyone who had already loaded the demo, and the failure is invisible because
 * everything still works, just with last week's file. The ETag the Hub's CDN
 * returns is a hash of the bytes, so putting it in the key means a changed file
 * misses the cache while an unchanged 900 MB of graphs still hits it.
 */

const CACHE_NAME = 'zerotts-weights-v1';

/** Query parameter carrying the ETag in a cache key. Never sent to the server:
 *  the key is a label for the Cache API, the fetch always uses the bare URL. */
const VERSION_PARAM = '__etag';

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
    const cache = await caches.open(CACHE_NAME);
    await purgeUnversioned(cache);
    return cache;
  } catch {
    // Private browsing or a blocked storage policy — still works, just slow
    // on every visit.
    return null;
  }
}

/** Done once per session; `null` until then. */
let purged: Promise<void> | null = null;

/**
 * Drop entries written by the pre-ETag scheme, once.
 *
 * Those were keyed by the bare `/resolve/main/` URL with nothing recording
 * WHICH revision they came from, so there is no way to tell a current copy from
 * a stale one — and a stale one is unfalsifiable: a re-fit voice is the same
 * size and still decodes, it just isn't the voice the Hub is serving. Anyone
 * carrying such a cache pays for one more full download; after that, an ETag
 * mismatch re-fetches exactly the files that changed and nothing else.
 */
function purgeUnversioned(cache: Cache): Promise<void> {
  purged ??= (async () => {
    for (const request of await cache.keys()) {
      if (!new URL(request.url).searchParams.has(VERSION_PARAM)) await cache.delete(request);
    }
  })();
  return purged;
}

interface HeadInfo { size: number; etag: string | null; }

/**
 * HEAD a URL, once per session.
 *
 * Memoized because the size note, the aggregate progress bar and the download
 * itself all want the same answer, and three HEADs per file for a six-file model
 * is three round-trips of nothing.
 */
const heads = new Map<string, Promise<HeadInfo | null>>();

function head(url: string): Promise<HeadInfo | null> {
  let info = heads.get(url);
  if (!info) {
    info = fetch(url, { method: 'HEAD' }).then(
      (r) => (r.ok ? { size: responseSize(r.headers), etag: r.headers.get('etag') } : null),
      () => null,  // offline, or CORS on a host that does not answer HEAD
    );
    heads.set(url, info);
  }
  return info;
}

/** The cache key for a URL at a given version. */
function versioned(url: string, etag: string | null): string {
  if (!etag) return url;
  const key = new URL(url);
  key.searchParams.set(VERSION_PARAM, etag);
  return key.toString();
}

/** True when `request` is some version of `url` — any version, or none. */
function isVersionOf(requestUrl: string, url: string): boolean {
  const stripped = new URL(requestUrl);
  stripped.searchParams.delete(VERSION_PARAM);
  return stripped.toString() === url;
}

/**
 * Any cached copy of `url`, whatever version it was stored under.
 *
 * The fallback for when the HEAD failed: without an ETag there is nothing to
 * compare, and serving a possibly-stale file beats failing to load the model at
 * all — which is what an offline reload would otherwise do.
 */
async function anyVersion(cache: Cache, url: string): Promise<Response | undefined> {
  for (const request of await cache.keys()) {
    if (isVersionOf(request.url, url)) return cache.match(request);
  }
  return undefined;
}

/** Drop every cached copy of `url` except the one under `keep`. */
async function evictOtherVersions(cache: Cache, url: string, keep: string): Promise<void> {
  for (const request of await cache.keys()) {
    if (request.url !== keep && isVersionOf(request.url, url)) await cache.delete(request);
  }
}

/** Fetch a URL as an ArrayBuffer, reporting progress and caching the result. */
export async function fetchWithCache(
  url: string,
  onProgress?: ProgressFn,
  overall = { loaded: 0, total: 0 },
): Promise<ArrayBuffer> {
  const cache = await openCache();
  const info = await head(url);
  const key = versioned(url, info?.etag ?? null);
  const hit = await (info?.etag ? cache?.match(key) : cache && anyVersion(cache, url));
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
    await store(cache, url, key, buf);
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
  await store(cache, url, key, out.slice(0).buffer);
  return out.buffer;
}

/** Cache a freshly downloaded file, retiring whatever version preceded it. */
async function store(
  cache: Cache | null, url: string, key: string, body: ArrayBuffer,
): Promise<void> {
  if (!cache) return;
  await cache.put(key, new Response(body));
  // Only after the new copy is safely in: an eviction that ran first would, if
  // the download failed, leave the user with neither.
  await evictOtherVersions(cache, url, key);
}

/**
 * Size of a response, preferring Hugging Face's `x-linked-size`.
 *
 * Big files on the Hub are Git LFS pointers served through a CDN redirect, and
 * the 302 describes the POINTER: a few hundred bytes for a 391 MB graph. A
 * browser follows the redirect and reports the CDN's `content-length`, which is
 * the real size — but a caller that sees the redirect itself (or a mirror that
 * serves pointers directly) needs `x-linked-size`, which HF puts on the 302.
 */
function responseSize(headers: Headers): number {
  return Number(headers.get('x-linked-size') ?? headers.get('content-length') ?? 0);
}

/** HEAD every URL to size the aggregate progress bar before downloading. */
export async function totalBytes(urls: string[]): Promise<number> {
  const infos = await Promise.all(urls.map(head));
  // An unknown size just makes the bar less precise.
  return infos.reduce((total, info) => total + (info?.size ?? 0), 0);
}

/** Whether every URL is already cached AT ITS CURRENT VERSION — which is what
 *  decides whether the UI promises a fast load or warns about a download. */
export async function isCached(urls: string[]): Promise<boolean> {
  const cache = await openCache();
  if (!cache) return false;
  const hits = await Promise.all(urls.map(async (url) => {
    const info = await head(url);
    return info?.etag ? cache.match(versioned(url, info.etag)) : anyVersion(cache, url);
  }));
  return hits.every(Boolean);
}

export async function clearCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}
