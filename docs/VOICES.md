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

## Using voices

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

tts.list_voices()                    # ['arya', ...]

v = tts.load_voice("arya")
v.emb.shape                          # (1, 10, 768)
v.language, v.description, v.preview_path

tts.synthesize("Xin chào.", voice="arya")   # by name
tts.synthesize("Xin chào.", voice=v)        # by object
tts.synthesize("Xin chào.", voice=v.emb)    # by raw array
tts.synthesize("Xin chào.", voice=None)     # unconditional
```

`cfg_scale > 1` sharpens the identity by guiding away from the unconditional
branch, at twice the per-frame cost:

```python
tts.synthesize("Xin chào.", voice="arya", cfg_scale=2.0)
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
    meta.json      # optional: {"name": "...", "language": "vi", "description": "..."}
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

### The query-count check is deliberate

Loading refuses a voice whose `n_voice_queries` does not match the model's:

```
ValueError: voice 'x' was built with n_voice_queries=8, but this model uses 10.
It belongs to different weights.
```

Latents built for a different model have the right dtype and rank, so they would
feed the graph cleanly and generate confident nonsense rather than failing. The
check is what turns a silent quality regression into an error.

## Reference audio is not published

Voice packs ship a short `preview.wav` but never the original reference clip. The
reference is a real person's recording *and* the input to the cloning step, so
publishing it would both distribute their voice and hand over the thing the
encoder was held back to protect.
