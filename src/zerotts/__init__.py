"""ZeroTTS — Vietnamese zero-shot text-to-speech, ONNX runtime, no PyTorch.

    from zerotts import ZeroTTS

    tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
    audio = tts.synthesize("Xin chào các bạn.", voice="maichi")
    tts.save_audio(audio, "out.wav")
"""

from .hub import DEFAULT_REPO_ID, resolve_model_dir
from .synthesizer import ZeroTTS
from .text_norm import normalize_vi_text
from .voices import Voice, list_voices, load_voice

__version__ = "0.1.1"

__all__ = [
    "ZeroTTS",
    "normalize_vi_text",
    "Voice",
    "list_voices",
    "load_voice",
    "resolve_model_dir",
    "DEFAULT_REPO_ID",
    "__version__",
]
