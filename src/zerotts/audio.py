"""Audio I/O helpers — plain numpy + soundfile/scipy, no torch."""

from __future__ import annotations

from math import gcd

import numpy as np

SAMPLE_RATE = 48_000


def load_wav_mono(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a wav, downmix to mono, resample to ``target_sr``. Returns (T,) float32."""
    import soundfile as sf

    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        from scipy.signal import resample_poly

        g = gcd(target_sr, sr)
        wav = resample_poly(wav, target_sr // g, sr // g)
    return wav.astype(np.float32)


def resample_wav(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample a 1-D float32 array, matching load_wav_mono's resampler."""
    if sr_from == sr_to:
        return wav.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(wav, sr_to // g, sr_from // g).astype(np.float32)


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a (..., T) float array (leading dims squeezed) as 16-bit PCM wav."""
    import soundfile as sf

    sf.write(path, np.asarray(audio).squeeze(), sample_rate, subtype="PCM_16")


def concat_with_silence(chunks: list[np.ndarray], silence_sec: float,
                        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Join (1, T) or (T,) chunks with ``silence_sec`` of silence between them."""
    parts = [np.asarray(c).reshape(-1) for c in chunks if np.asarray(c).size]
    if not parts:
        return np.zeros((1, 0), dtype=np.float32)
    if silence_sec > 0 and len(parts) > 1:
        gap = np.zeros(int(silence_sec * sample_rate), dtype=np.float32)
        joined: list = []
        for i, p in enumerate(parts):
            if i:
                joined.append(gap)
            joined.append(p)
        parts = joined
    return np.concatenate(parts).astype(np.float32)[None, :]
