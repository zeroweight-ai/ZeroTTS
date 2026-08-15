"""WER / SSIM / UTMOS / silence scorers for the ZeroTTS benchmark.

**These need PyTorch** — they are torch models (PhoWhisper, WavLM-SV, UTMOSv2).
That is why they live behind ``pip install "zerotts[eval]"`` and not in the
`zerotts` package itself: measuring quality needs torch, generating audio does
not.

    WER    vinai/PhoWhisper-large transcript of the GENERATED audio vs. the text
           the model was asked to say. A Vietnamese-specialized Whisper finetune,
           NOT raw whisper-large-v3 — scoring Vietnamese TTS with raw Whisper
           measures the ASR's weakness as much as the TTS system's, and
           compresses the differences between models.
    SSIM   cosine similarity of microsoft/wavlm-base-plus-sv x-vectors between
           the generated audio and the reference clip
    UTMOS  UTMOSv2 naturalness MOS of the generated audio
    SIL    excess leading/trailing/mid-utterance silence, in seconds. Nothing in
           WER/SSIM/UTMOS penalizes dead air — an ASR transcribes a clip with two
           seconds of leading silence perfectly — so it is measured separately.

Multi-reference WER
───────────────────
``score_wer`` takes a LIST of acceptable references and returns the MINIMUM WER
over them. The benchmark ships both the written text ("Lạm phát năm nay ở mức
3,2%.") and a hand-curated spoken-out normalization ("... ba phẩy hai phần
trăm."), because the ASR is free to emit either and that choice is the ASR's
formatting policy, not the TTS model's. ``--use_vinorm`` adds a third reference
from soe-vinorm's automatic normalization; see docs/BENCHMARKS.md for why that
column exists.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

DEFAULT_ASR = "vinai/PhoWhisper-large"
DEFAULT_SV = "microsoft/wavlm-base-plus-sv"


# ── text normalization + edit distance ───────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_for_scoring(text: str) -> str:
    """NFC, lowercase, strip punctuation, collapse whitespace.

    WER should measure whether the words came out, not whether the ASR chose to
    write a comma.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_error_rate(hyp: str, ref: str) -> float:
    ref_words = normalize_for_scoring(ref).split()
    hyp_words = normalize_for_scoring(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def score_wer(transcript: str, references: list) -> tuple:
    """Minimum WER over the accepted references. Returns (wer, index)."""
    if not references:
        return float("nan"), -1
    scores = [word_error_rate(transcript, r) for r in references]
    best = int(np.argmin(scores))
    return float(scores[best]), best


# ── resampling ───────────────────────────────────────────────────────────────

def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16_000:
        return wav.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, 16_000)
    return resample_poly(wav, 16_000 // g, sr // g).astype(np.float32)


# ── silence ──────────────────────────────────────────────────────────────────

def excess_silence(wav: np.ndarray, sr: int, top_db: float = 35.0,
                   max_lead: float = 0.1, max_tail: float = 0.1,
                   max_pause: float = 0.6) -> float:
    """Seconds of silence beyond what natural speech justifies: leading and
    trailing silence past a small allowance, plus any interior pause longer than
    ``max_pause``."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    frame, hop = 1024, 256
    if len(wav) < frame:
        return 0.0
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(wav[idx].astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return len(wav) / sr
    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / peak)
    voiced = db > -top_db
    if not voiced.any():
        return len(wav) / sr

    sec_per_frame = hop / sr
    first, last = int(np.argmax(voiced)), int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    total = max(0.0, first * sec_per_frame - max_lead)
    total += max(0.0, (len(voiced) - 1 - last) * sec_per_frame - max_tail)

    run = 0
    for v in voiced[first:last + 1]:
        if v:
            if run * sec_per_frame > max_pause:
                total += run * sec_per_frame - max_pause
            run = 0
        else:
            run += 1
    return float(total)


# ── model-backed scorers ─────────────────────────────────────────────────────

class ASRScorer:
    """PhoWhisper (or any Whisper-architecture checkpoint) transcription."""

    def __init__(self, model_id: str = DEFAULT_ASR, device: str = "cuda"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def transcribe(self, wav16: np.ndarray, lang: str = "vi") -> str:
        import torch

        feats = self.processor(wav16, sampling_rate=16_000,
                               return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=self.model.dtype)
        with torch.no_grad():
            ids = self.model.generate(feats, language=lang, task="transcribe",
                                      max_new_tokens=256)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class SSIMScorer:
    """WavLM-base-plus-sv x-vector cosine similarity."""

    def __init__(self, model_id: str = DEFAULT_SV, device: str = "cuda"):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device if torch.cuda.is_available() else "cpu"
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(self.device).eval()

    def embed(self, wav16: np.ndarray):
        import torch

        inputs = self.extractor(wav16, sampling_rate=16_000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.model(**inputs).embeddings[0].cpu().numpy()

    def score(self, gen16: np.ndarray, ref16: np.ndarray) -> float:
        a, b = self.embed(gen16), self.embed(ref16)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class UTMOSScorer:
    """UTMOSv2 naturalness MOS. Optional — returns NaN if unavailable."""

    def __init__(self, device: str = "cuda"):
        self.model = None
        try:
            import utmosv2

            self.model = utmosv2.create_model(pretrained=True)
        except Exception as exc:  # pragma: no cover
            print(f"[scorers] UTMOSv2 unavailable ({exc}); UTMOS will be NaN")

    def score(self, wav_path: str) -> float:
        if self.model is None:
            return float("nan")
        try:
            return float(self.model.predict(input_path=wav_path))
        except Exception:
            return float("nan")
