"""Precomputed voice packs.

A "voice" in ZeroTTS is a small array of speaker latents — shape
``(1, n_voice_queries, d_model)`` float32 — that gets prepended to the sequence.
That array is the *entire* speaker conditioning; there is no reference audio, no
transcript, and no prompt frames involved at generation time.

Those latents are produced by a voice encoder that reads a reference clip. **The
encoder is not part of this release**, so this package cannot create a voice
from a wav — it can only load ones that already exist. See the README, or
zeroweight.ai, for how to obtain latents for your own speaker.

That boundary is narrower than it sounds: a voice pack is just a .npz, so
latents obtained elsewhere drop into ``voices/<name>/voice.npz`` and work with
no code change.

Layout of a voice pack directory:

    voices/
      index.json                 # optional manifest: name, display_name, tags...
      <name>/
        voice.npz                # required: {n_voice_queries: int64, voice_emb: (1,Q,D) f32}
        voice.bin                # optional: raw f32 of voice_emb, for the JS demo
        preview.wav              # optional
        meta.json                # optional: display_name, gender, tags, description...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Voice:
    """A loaded voice pack."""

    name: str
    emb: np.ndarray            # (1, n_voice_queries, d_model) float32
    meta: dict
    preview_path: str | None = None

    @property
    def n_voice_queries(self) -> int:
        return int(self.emb.shape[1])

    @property
    def language(self) -> str:
        return str(self.meta.get("language", "vi"))

    @property
    def description(self) -> str:
        return str(self.meta.get("description", ""))

    @property
    def display_name(self) -> str:
        """Human-readable name, e.g. "Mai Chi" for the pack directory "maichi"."""
        return str(self.meta.get("display_name", self.name))

    @property
    def gender(self) -> str:
        return str(self.meta.get("gender", ""))

    @property
    def tags(self) -> list[str]:
        """Free-form labels — gender, age, register, tone ("nữ", "trẻ", "kể
        chuyện", "ấm áp"...) — for filtering or displaying a voice picker."""
        return list(self.meta.get("tags", []))


def _voice_dirs(voices_root: Path):
    if not voices_root.is_dir():
        return []
    return sorted(d for d in voices_root.iterdir() if d.is_dir() and (d / "voice.npz").exists())


def list_voices(voices_root: str | Path) -> list:
    """Voice names available under ``voices_root``, sorted."""
    return [d.name for d in _voice_dirs(Path(voices_root))]


def load_voice(voices_root: str | Path, name: str, expect_queries: int | None = None) -> Voice:
    """Load one voice pack by name.

    ``expect_queries`` is the model's ``n_voice_queries``. A mismatch is fatal
    and says so: latents built for a different model would still be the right
    dtype and rank, so they would feed the graph cleanly and produce confident
    nonsense.
    """
    root = Path(voices_root)
    vdir = root / name
    npz = vdir / "voice.npz"
    if not npz.exists():
        available = list_voices(root)
        raise FileNotFoundError(
            f"no voice {name!r} in {root} (available: {available or 'none'})")

    data = np.load(npz)
    emb = np.asarray(data["voice_emb"], dtype=np.float32)
    if emb.ndim == 2:
        emb = emb[None, :, :]

    stored_q = int(data["n_voice_queries"]) if "n_voice_queries" in data else int(emb.shape[1])
    if stored_q != emb.shape[1]:
        raise ValueError(
            f"voice {name!r} is inconsistent: n_voice_queries={stored_q} but "
            f"voice_emb has {emb.shape[1]} queries.")
    if expect_queries is not None and stored_q != expect_queries:
        raise ValueError(
            f"voice {name!r} was built with n_voice_queries={stored_q}, but this "
            f"model uses {expect_queries}. It belongs to different weights.")

    meta_path = vdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    preview = vdir / "preview.wav"
    return Voice(name=name, emb=emb, meta=meta,
                 preview_path=str(preview) if preview.exists() else None)


def load_index(voices_root: str | Path) -> dict:
    """The ``index.json`` manifest, or a minimal one synthesized from the dirs."""
    root = Path(voices_root)
    index_path = root / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {"voices": [{"name": n} for n in list_voices(root)]}
