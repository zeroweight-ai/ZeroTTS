#!/usr/bin/env bash
# Build the browser runtime and drop it where Vite can serve it.
#
# Needs the Emscripten SDK on PATH (source ~/emsdk/emsdk_env.sh). The two
# artifacts land in js/public/ggml/, which Vite copies to the bundle root
# verbatim — the .js locates its .wasm and spawns its pthread workers from
# its own URL, so the pair must stay side by side under that name.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v emcmake >/dev/null; then
    echo "emcmake not found — source your emsdk_env.sh first" >&2
    exit 1
fi

emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release .
cmake --build build-wasm -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

mkdir -p ../js/public/ggml
cp build-wasm/zerotts-wasm.js build-wasm/zerotts-wasm.wasm ../js/public/ggml/

# bench-ggml.html fetches GGUFs from /ggml/models/. A relative symlink keeps the
# (large, gitignored) weights in cpp/models rather than copying them into the
# served tree. Vite's dev server follows it; both ends of the link are ignored.
ln -sfn ../../../cpp/models ../js/public/ggml/models

ls -la ../js/public/ggml/
