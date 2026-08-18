import { defineConfig } from 'vite';

export default defineConfig({
  // onnxruntime-web ships .wasm/.mjs assets that must not be inlined or renamed.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', assetsInlineLimit: 0 },
  // src/worker.ts is where the model runs; it must be a real module worker so
  // its (large) onnxruntime-web graph stays off the page's bundle.
  worker: { format: 'es' },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  server: {
    // src/loader.ts imports ../../webui/test_samples.txt?raw so both demos read
    // the same sample file; without this, Vite's dev server refuses to serve it.
    fs: { allow: ['..'] },
    headers: {
      // Required for SharedArrayBuffer, which onnxruntime-web needs for
      // multi-threaded WASM. Without these it silently falls back to a single
      // thread and generation is several times slower.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
