"""Pluggable TTS backends for the benchmark runner.

A system under evaluation is a subclass of :class:`TTSModel` with a ``@register``
decorator; the runner only ever calls ``synthesize(TTSRequest) -> np.ndarray`` and
never learns which backend it holds. Adding a system is a subclass, not a fork of
the runner.

    --model zerotts:zeroweight-ai/ZeroTTS
    --model zerotts:/path/to/local/model_dir
    --model xtts:thivux/XTTS-v2-vietnamse
    --model xtts:capleaf/viXTTS
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TTSRequest:
    text: str
    ref_audio: str            # path to the reference clip
    language: str = "vi"
    subset: str = ""
    item_id: str = ""
    extra: dict = field(default_factory=dict)


class TTSModel(ABC):
    name: str = "tts"
    sample_rate: int = 24_000

    @abstractmethod
    def synthesize(self, req: TTSRequest) -> np.ndarray:
        """Mono float32 in [-1, 1] at ``self.sample_rate``.

        Return an empty array for a failed generation; the runner counts it and
        moves on rather than killing the sweep.
        """

    def config(self) -> dict:
        return {}


REGISTRY: dict = {}


def register(backend: str):
    def deco(cls):
        REGISTRY[backend] = cls
        return cls
    return deco


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_model_args(pairs) -> dict:
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--model_arg expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = _coerce(value.strip())
    return out


def build_model(spec: str, device: str = "cuda", **kwargs) -> TTSModel:
    backend, _, target = spec.partition(":")
    if backend not in REGISTRY:
        raise SystemExit(
            f"unknown backend {backend!r}. Registered: {sorted(REGISTRY)}.")
    return REGISTRY[backend](target or None, device=device, **kwargs)


# ── ZeroTTS ──────────────────────────────────────────────────────────────────

@register("zerotts")
class ZeroTTSModel(TTSModel):
    """ZeroTTS through the public package.

    Note what is NOT happening here: the reference clip is not encoded. This
    release has no voice encoder, so the benchmark drives ZeroTTS with the
    *precomputed voice pack* named by the item (``extra['voice']``) when the
    model ships one, and the unconditional prefix otherwise.

    SSIM is still computed against the item's reference clip by the runner, which
    is why the benchmark's `voice` field must name the same speaker the reference
    belongs to — otherwise SSIM measures the wrong thing.
    """

    name = "zerotts"

    def __init__(self, target: str | None, device: str = "cpu", **kwargs):
        from zerotts import ZeroTTS

        self.model_id = target or "zeroweight-ai/ZeroTTS"
        self.tts = ZeroTTS.from_pretrained(
            self.model_id, intra_op_num_threads=int(kwargs.pop("threads", 8)))
        self.sample_rate = self.tts.sample_rate

        self.cfg_scale = float(kwargs.pop("cfg_scale", 1.0))
        self.audio_temperature = float(kwargs.pop("audio_temperature", 0.8))
        self.audio_topk = int(kwargs.pop("audio_topk", 25))
        self.audio_topp = float(kwargs.pop("audio_topp", 0.95))
        self.audio_repetition_penalty = float(
            kwargs.pop("audio_repetition_penalty", 1.2))
        self.max_frames = int(kwargs.pop("max_frames", 1500))
        self.eoa_extra_frames = int(kwargs.pop("eoa_extra_frames", 1))
        self._available = set(self.tts.list_voices())

    def synthesize(self, req: TTSRequest) -> np.ndarray:
        voice = req.extra.get("voice")
        if voice not in self._available:
            voice = None
        audio = self.tts.synthesize(
            req.text, voice=voice, cfg_scale=self.cfg_scale,
            audio_temperature=self.audio_temperature, audio_topk=self.audio_topk,
            audio_topp=self.audio_topp,
            audio_repetition_penalty=self.audio_repetition_penalty,
            max_frames=self.max_frames, eoa_extra_frames=self.eoa_extra_frames)
        return np.asarray(audio).reshape(-1)

    def config(self) -> dict:
        return {
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "cfg_scale": self.cfg_scale,
            "audio_temperature": self.audio_temperature,
            "audio_topk": self.audio_topk,
            "audio_topp": self.audio_topp,
            "audio_repetition_penalty": self.audio_repetition_penalty,
            "max_frames": self.max_frames,
            "eoa_extra_frames": self.eoa_extra_frames,
            "voices_available": sorted(self._available),
        }


# ── Coqui XTTS-v2 Vietnamese finetunes (baselines) ───────────────────────────

def _patch_isin_mps_friendly() -> None:
    """`coqui-tts` calls transformers' `isin_mps_friendly`, removed in recent
    versions. Restore a shim so the baselines run at all under a modern stack."""
    try:
        import transformers.pytorch_utils as pu

        if not hasattr(pu, "isin_mps_friendly"):
            import torch

            def isin_mps_friendly(elements, test_elements):
                return torch.isin(elements, test_elements)

            pu.isin_mps_friendly = isin_mps_friendly
    except Exception:
        pass


def _patch_xtts_vi_tokenizer() -> None:
    """XTTS's stock tokenizer has no Vietnamese branch: `preprocess_text` raises
    on `lang="vi"`. Register one that lowercases and collapses whitespace and
    nothing else.

    This is deliberately minimal. It notably does NOT expand numbers, dates or
    abbreviations — XTTS has no such rules for Vietnamese, and inventing them
    here would mean benchmarking our text frontend rather than their model. The
    benchmark instead reports a `+vinorm` WER column that supplies normalized
    text as an accepted *reference*, which measures the same question without
    modifying either system. See docs/BENCHMARKS.md.
    """
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, collapse_whitespace, lowercase

    original = VoiceBpeTokenizer.preprocess_text

    def preprocess_text(self, txt, lang):
        if lang in ("vi", "vi-vn"):
            return collapse_whitespace(lowercase(txt))
        return original(self, txt, lang)

    VoiceBpeTokenizer.preprocess_text = preprocess_text


@register("xtts")
class XTTSModel(TTSModel):
    """Coqui XTTS-v2 finetunes. Needs `pip install coqui-tts`."""

    name = "xtts"
    sample_rate = 24_000

    def __init__(self, target: str | None, device: str = "cuda", **kwargs):
        from pathlib import Path

        import torch

        if not target:
            raise SystemExit("xtts needs --model xtts:<hf-repo-or-local-dir>")

        _patch_isin_mps_friendly()
        _patch_xtts_vi_tokenizer()

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self.name = f"xtts-{Path(target).name}"
        self.device = device if torch.cuda.is_available() else "cpu"
        paths = self._resolve(target)

        config = XttsConfig()
        config.load_json(paths["config"])
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=paths["dir"],
                                   use_deepspeed=False)
        self.model.to(self.device).eval()
        self.target = target

    @staticmethod
    def _resolve(target: str) -> dict:
        from pathlib import Path

        local = Path(target)
        if local.is_dir():
            return {"dir": str(local), "config": str(local / "config.json")}

        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(target))
        return {"dir": str(d), "config": str(d / "config.json")}

    def synthesize(self, req: TTSRequest) -> np.ndarray:
        import torch

        with torch.no_grad():
            gpt_latent, speaker_emb = self.model.get_conditioning_latents(
                audio_path=[req.ref_audio])
            out = self.model.inference(
                text=req.text, language="vi",
                gpt_cond_latent=gpt_latent, speaker_embedding=speaker_emb)
        return np.asarray(out["wav"], dtype=np.float32).reshape(-1)

    def config(self) -> dict:
        return {"target": self.target, "sample_rate": self.sample_rate,
                "device": self.device}
