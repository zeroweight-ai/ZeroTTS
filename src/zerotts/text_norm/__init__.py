"""Vietnamese text normalization for TTS input.

Pure stdlib regex — no torch, onnx, sklearn, or network access — so it can sit
in front of the ONNX-only inference path without changing what that path
depends on. See vi_normalizer.py for the exact scope and for what is
deliberately left alone.
"""

from .vi_normalizer import expand_number, load_abbreviations, normalize_vi_text

__all__ = ["normalize_vi_text", "expand_number", "load_abbreviations"]
