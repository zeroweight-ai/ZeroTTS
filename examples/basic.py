"""Synthesize one sentence to a wav file."""

from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
print("voices:", tts.list_voices())

audio = tts.synthesize("Xin chào các bạn, mình là ZeroTTS.", voice="maichi")
tts.save_audio(audio, "basic.wav")
print(f"wrote basic.wav ({audio.shape[-1] / tts.sample_rate:.2f}s)")
