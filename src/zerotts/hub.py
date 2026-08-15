"""Resolving a ZeroTTS model directory — local path or Hugging Face repo.

Published layout (hf.co/zeroweight-ai/ZeroTTS):

    config.json
    tokenizer.json
    null_voice_emb.npy
    onnx/{text_encoder,prefix_step,local_frame_decode}.onnx
    onnx/codec/...            # vendored MOSS decoder, Apache-2.0
    voices/index.json
    voices/<name>/{voice.npz,voice.bin,preview.wav,meta.json}

``voice.bin`` exists only for the browser demo (raw f32, no zip parser needed),
so the Python download skips it — it is a byte-for-byte duplicate of what
``voice.npz`` already carries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_REPO_ID = "zeroweight-ai/ZeroTTS"

# voice.bin is browser-only and duplicates voice.npz; never worth the bytes here.
_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "null_voice_emb.npy",
    # The codec's canonical silence frame, used to pad between segments in
    # long-form streaming. Small, and there is no way to derive it without the
    # codec encoder, which is not shipped.
    "silence_frame.npy",
    "onnx/*",
    "onnx/codec/*",
    "voices/index.json",
    "voices/*/voice.npz",
    "voices/*/meta.json",
    "voices/*/preview.wav",
]

REQUIRED_FILES = (
    "config.json",
    "null_voice_emb.npy",
    "onnx/text_encoder.onnx",
    "onnx/prefix_step.onnx",
    "onnx/local_frame_decode.onnx",
)


def resolve_model_dir(
    model_id: str | Path = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """A local directory containing the model, downloading it if needed.

    ``model_id`` is either an existing local directory (used as-is, nothing is
    fetched) or a Hugging Face repo id.
    """
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=str(model_id),
        revision=revision,
        cache_dir=cache_dir or os.environ.get("HF_HOME_MODELS"),
        local_files_only=local_files_only,
        allow_patterns=_ALLOW_PATTERNS,
    ))


def load_config(model_dir: str | Path) -> dict:
    """Read and validate ``config.json``, checking the graphs are actually there."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{model_dir} has no config.json — it is not a ZeroTTS model directory.")

    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{model_dir} is missing {missing}. If this is a local export, it may "
            f"predate the published layout (graphs under onnx/, text_encoder.onnx "
            f"formerly named char_embed.onnx).")

    return json.loads(config_path.read_text())
