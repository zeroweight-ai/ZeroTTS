import { defineConfig } from 'vite';

export default defineConfig({
  // onnxruntime-web ships .wasm/.mjs assets that must not be inlined or renamed.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', assetsInlineLimit: 0 },
  server: {
    headers: {
      // Required for SharedArrayBuffer, which onnxruntime-web needs for
      // multi-threaded WASM. Without these it silently falls back to a single
      // thread and generation is several times slower.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
