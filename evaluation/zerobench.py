"""Locate the ZeroBench-TTS scorer and make it importable.

Evaluation lives in the benchmark, not here. ``zerobench_eval`` is published
inside the `ZeroBench-TTS
<https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS>`_ dataset repo and
is the single source of truth for how a number is produced — the ASR pair, the
reference-expansion policy, the aggregation.

Vendoring a copy here would let ZeroTTS's published numbers drift from the
benchmark's, which is exactly the claim a benchmark should not have to make on
its own behalf. Fetching it also means *you* score ZeroTTS with the same code
that scores every other system.

Resolution order:

1. already importable (``pip install``-ed, or on ``PYTHONPATH``)
2. ``$ZEROBENCH_DIR`` — a local clone of the dataset repo
3. downloaded from the Hub and cached (code + metadata + reference audio)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = "zeroweight-ai/ZeroBench-TTS"

#: Everything the scorer needs; the parquet configs are not required.
_PATTERNS = ["zerobench_eval/*", "metadata.jsonl", "voices.jsonl", "audio/*"]


def ensure_zerobench(local_dir: str | None = None) -> Path:
    """Return the directory containing ``zerobench_eval/``, importable on exit."""
    if local_dir:
        root = Path(local_dir).resolve()
        if not (root / "zerobench_eval").is_dir():
            raise SystemExit(f"{root} has no zerobench_eval/ — not a ZeroBench-TTS clone")
        return _add_to_path(root)

    try:                                        # 1. already available
        import zerobench_eval  # noqa: F401
        return Path(zerobench_eval.__file__).resolve().parent.parent
    except ImportError:
        pass

    env = os.environ.get("ZEROBENCH_DIR")       # 2. local clone
    if env and (Path(env) / "zerobench_eval").is_dir():
        return _add_to_path(Path(env).resolve())

    try:                                        # 3. the Hub
        from huggingface_hub import snapshot_download
    except ImportError as e:                    # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required to fetch the scorer — "
            'pip install "zerotts[eval]"') from e

    print(f"[zerobench] fetching the scorer from {REPO_ID} ...", flush=True)
    return _add_to_path(Path(snapshot_download(
        REPO_ID, repo_type="dataset", allow_patterns=_PATTERNS)))


def _add_to_path(root: Path) -> Path:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
