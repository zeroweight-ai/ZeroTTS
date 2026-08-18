# Voices

## What a voice is

A voice is a small float32 array of speaker latents:

```
voice_emb : (1, n_voice_queries, d_model)     # e.g. (1, 10, 768) — ~30 KB
```

That array is the **entire** speaker conditioning. At generation time it is
prepended to the sequence and nothing else about the speaker is involved: no
reference transcript, no in-context audio prompt, no teacher-forced frames. Two
runs with the same latents and the same seed produce the same speaker.

With no voice at all, the model falls back to a learned *unconditional* prefix
(`null_voice_emb.npy`). That is a valid mode — it produces natural speech — but
the identity is whatever the model picks and is not stable across runs.

## Built-in voices

Eight presets ship with the weights repo, each tagged by gender, age and
register/tone so you can pick one by ear or by filter. `maichi` (Mai Chi) is
the default used throughout this README and the demos.

| id | name | gender | tags |
|---|---|---|---|
| `maichi` | Mai Chi | nữ | trẻ · kể chuyện · nhẹ nhàng · thân thiện |
| `baotrang` | Bảo Trang | nữ | trưởng thành · tin tức · rõ ràng · trung tính |
| `kimoanh` | Kim Oanh | nữ | trung niên · kể chuyện · ấm áp · truyền cảm |
| `hamy` | Hà My | nữ | trẻ · hoạt hình · cao · biểu cảm |
| `giahuy` | Gia Huy | nam | trẻ · kể chuyện · trầm ấm · tâm tình |
| `huuduc` | Hữu Đức | nam | lớn tuổi · kể chuyện · trầm · điềm đạm |
| `quangminh` | Quang Minh | nam | trẻ · tin tức · rõ ràng · dứt khoát |
| `tiendat` | Tiến Đạt | nam | trẻ · bình luận · sôi nổi · năng lượng cao |

`v.tags` (a `list[str]`) and `v.gender`/`v.display_name` read straight off each
pack's `meta.json` — see [Voice pack format](#voice-pack-format) below.

## Using voices

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

tts.list_voices()                    # ['maichi', 'baotrang', ...]

v = tts.load_voice("maichi")
v.emb.shape                          # (1, 10, 768)
v.language, v.description, v.preview_path

tts.synthesize("Xin chào.", voice="maichi")   # by name
tts.synthesize("Xin chào.", voice=v)        # by object
tts.synthesize("Xin chào.", voice=v.emb)    # by raw array
tts.synthesize("Xin chào.", voice=None)     # unconditional
```

`cfg_scale > 1` sharpens the identity by guiding away from the unconditional
branch, at twice the per-frame cost:

```python
tts.synthesize("Xin chào.", voice="maichi", cfg_scale=2.0)
```

## Cloning your own voice

**Not available in this release.** The latents above are produced by a voice
encoder that reads a reference clip, and that encoder is not published. This
package can load voices; it cannot create them from audio. There is no flag,
environment variable, or optional dependency that enables it — the code path does
not exist in the package, and the graph is not in the weights repo.

To get latents for your own speaker, visit **[zeroweight.ai](https://zeroweight.ai)**
or get in touch.

Because a voice is just an array, latents obtained that way are a drop-in:

```
voices/
  my-speaker/
    voice.npz      # np.savez(f, n_voice_queries=np.int64(10), voice_emb=emb)
    meta.json      # optional: {"name": ..., "display_name": ..., "language": "vi",
                   #            "gender": ..., "tags": [...], "description": ...}
    preview.wav    # optional
```

Point `ZeroTTS.from_pretrained` at a local directory laid out like the published
repo, or pass the array directly with `voice=emb`.

## Voice pack format

```
voices/
  index.json                     # manifest of every voice
  <name>/
    voice.npz                    # required
    voice.bin                    # optional: raw f32, for the browser demo
    preview.wav                  # optional
    meta.json                    # optional
```

`voice.npz` holds exactly two arrays:

| key | dtype | shape |
|---|---|---|
| `n_voice_queries` | int64 | scalar |
| `voice_emb` | float32 | `(1, n_voice_queries, d_model)` |

`voice.bin` is the same `voice_emb` as raw little-endian float32 in C order. It
exists only so the browser demo can `fetch` + `new Float32Array` without a zip
parser; the Python loader ignores it and `from_pretrained` does not download it.