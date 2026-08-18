/**
 * Sample texts for the demo's picker.
 *
 * Imported as a raw string from the same webui/test_samples.txt the Python UI
 * reads, so the two demos stay in step and the file has one home. Vite inlines
 * it at build time (see vite.config.ts's `server.fs.allow`, which lets the
 * import reach outside js/).
 *
 * Kept out of loader.ts so the page can show the templates without importing
 * onnxruntime-web.
 */

import { parseSamples } from './chunking';

export async function loadSampleTexts(): Promise<Record<string, string>> {
  try {
    const raw = await import('../../webui/test_samples.txt?raw');
    return parseSamples((raw as { default: string }).default);
  } catch {
    return {};
  }
}
