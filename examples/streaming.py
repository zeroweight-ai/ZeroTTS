"""Streaming synthesis: measure time-to-first-audio, then play as it arrives.

The chunk schedule ramps — the first chunk is one frame (80 ms of audio) so
playback can start almost immediately, then chunks double up to 16 frames so the
per-call codec overhead is amortized once a playback buffer exists.
"""

import time

import numpy as np

from zerotts import ZeroTTS

TEXT = ("Đây là chế độ phát trực tuyến. Âm thanh được tạo ra và phát ngay lập tức, "
        "không cần chờ toàn bộ câu hoàn thành.")

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

t0 = time.perf_counter()
first = None
chunks = []
for chunk in tts.synthesize_stream(TEXT, voice="maichi"):
    if first is None:
        first = time.perf_counter() - t0
        print(f"time to first audio: {first * 1000:.0f} ms")
    chunks.append(chunk.reshape(-1))

audio = np.concatenate(chunks)
elapsed = time.perf_counter() - t0
dur = len(audio) / tts.sample_rate
print(f"{dur:.2f}s audio in {elapsed:.2f}s — {dur / elapsed:.1f}x realtime")

tts.save_audio(audio, "streaming.wav")
