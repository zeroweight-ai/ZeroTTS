"""Long-form text: segment it, synthesize each segment, join with short gaps.

The model is trained on utterances, not paragraphs, so a whole article has to be
split. `chunk_text` prefers to break on sentence punctuation, then commas, then
word boundaries, so no segment exceeds the budget however the input is (or is
not) punctuated. `normalize_punctuation` first rewrites written-only marks the
model has no prosody for — a newline becomes a sentence stop rather than
silently vanishing into a single space.
"""

from zerotts import ZeroTTS
from zerotts.audio import concat_with_silence
from zerotts.chunking import chunk_text, clean_segment_punctuation, normalize_punctuation

TEXT = """
Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc mỗi ngày.

Từ việc soạn thảo văn bản đến phân tích dữ liệu, các mô hình ngôn ngữ lớn đã trở
thành công cụ quen thuộc; nhiều người dùng chúng hàng giờ mà không nhận ra.
Câu hỏi đặt ra là: chúng ta sẽ dùng chúng như thế nào cho có trách nhiệm?
"""

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

segments = [clean_segment_punctuation(s)
            for s in chunk_text(normalize_punctuation(TEXT), max_chunk_sec=15.0)]
segments = [s for s in segments if s]

print(f"{len(segments)} segment(s)")
audio_chunks = []
for i, seg in enumerate(segments, 1):
    print(f"  [{i}/{len(segments)}] {seg}")
    audio_chunks.append(tts.synthesize(seg, voice="maichi"))

audio = concat_with_silence(audio_chunks, silence_sec=0.15, sample_rate=tts.sample_rate)
tts.save_audio(audio, "long_form.wav")
print(f"wrote long_form.wav ({audio.shape[-1] / tts.sample_rate:.2f}s)")
