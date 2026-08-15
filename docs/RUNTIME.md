# Runtime contract

What the three ONNX graphs expect and return, so the pipeline can be
reimplemented in another language (this is exactly what [`js/`](../js/) does).

This describes the **runtime**, not the model. Layer internals, training, and the
export script are not part of this release.

## Shapes at a glance

`D` = `d_model`, `K` = `num_codebooks` (16), `V` = `n_voice_queries` (10),
`C` = `codebook_size` (1024), `L` = text length, `B` = batch (1, or 2 with CFG).
All from `config.json`.

### `text_encoder.onnx` — once per utterance

| | name | shape | dtype |
|---|---|---|---|
| in | `text_ids` | `(B, L)` | int64 |
| in | `txt_lengths` | `(B,)` | int64 |
| out | `text_states` | `(B, L, D)` | float32 |
| out | `text_valid` | `(B, L)` | bool |
| out | `soa_embed` | `(B, 1, D)` | float32 |

`text_states` / `text_valid` are then passed into **every** `prefix_step` call —
the graph cross-attends to them, so they must be kept alive for the whole
utterance rather than consumed once.

(The file was called `char_embed.onnx` in older exports. Same graph.)

### `prefix_step.onnx` — once, then once per frame

One graph, two call shapes. Nothing in it branches on the *values* of `T` or
`S_past`, so a single traced graph replayed at different shapes is correct for
both.

| name | cold start | frame step | dtype |
|---|---|---|---|
| `external_embed` | `(B, V+1, D)` = `[voice ‖ soa_embed]` | `(B, 1, D)` zeros | float32 |
| `use_external_embed` | `(B, V+1)` all true | `(B, 1)` all false | bool |
| `frame_codes` | `(B, V+1, K)` zeros | `(B, 1, K)` sampled codes | int64 |
| `new_pos` | `[0 … V]` | `[V + 1 + t]` | int64 |
| `new_valid` | `(B, V+1)` true | `(B, 1)` true | bool |
| `new_bidirectional` | `[true×V, false]` | `(B, 1)` false | bool |
| `packed_kv` | `(n_layers, 2, B, n_heads, 0, d_head)` | previous output | float32 |
| `past_valid` | `(B, 0)` | previous `full_valid` | bool |
| `text_states`, `text_valid` | from the text encoder | same | float32 / bool |

Outputs: `hidden (B, T, D)` — take `[:, -1, :]` — plus the new `packed_kv` and
`full_valid`, which feed straight back in next call.

**Position ids are the easy thing to get wrong.** The voice block occupies logical
positions `0 … V-1`, `<soa>` sits at `V`, and audio frame `t` sits at `V + 1 + t`.
An off-by-one here does not raise — it silently offsets every RoPE position and
the output degrades in a way that looks like a bad checkpoint.

### `local_frame_decode.onnx` — once per frame

Decodes one entire frame: the control channel plus all `K` codebooks, with
embeddings, CFG mixing and sampling fused into the graph.

| | name | shape | dtype |
|---|---|---|---|
| in | `global_hidden` | `(B, D)` | float32 |
| in | `forbid_eoa` | `(1,)` | bool |
| in | `text_temperature`, `audio_temperature`, `audio_topp`, `audio_repetition_penalty`, `cfg_scale` | `(1,)` | float32 |
| in | `text_topk`, `audio_topk` | `(1,)` | int64 |
| in | `seen_mask` | `(1, K, C)` | bool |
| in | `ctrl_random_u` | `(1,)` | float32 |
| in | `audio_random_u` | `(1, K)` | float32 |
| out | `is_eoa` | `(1,)` | bool |
| out | `codes` | `(1, K)` | int64 |

Everything except `global_hidden` stays batch 1 even under CFG: the graph samples
once regardless.

`seen_mask` is the repetition-penalty history. A graph cannot carry
variable-length state between calls, so it is a dense boolean array standing in
for a per-codebook set — the caller must set `seen_mask[0, c, codes[0, c]] = True`
after every sampled frame, and reset it to all-false per utterance.

`ctrl_random_u` / `audio_random_u` are the sampler's random draws, supplied from
outside. That is what makes generation reproducible from a seed, and it lets a
port be tested for bit-exact agreement against the Python runtime.

## The loop

```
text_states, text_valid, soa = text_encoder(text_ids, txt_lengths)
h, kv, valid                 = prefix_step(cold start with [voice ‖ soa])

seen  = zeros(1, K, C, bool)
tail  = None
for t in 0, 1, 2, …:
    is_eoa, codes = local_frame_decode(h, forbid_eoa=(t < min_frames or tail is not None), …)
    seen[0, c, codes[0, c]] = True                      # for every c

    if tail is None and is_eoa:  tail = eoa_extra_frames
    if (tail is not None and tail <= 0) or t >= max_frames:  break

    emit codes
    if tail is not None:
        tail -= 1
        if tail <= 0: break

    h, kv, valid = prefix_step(frame step with codes, pos = V + 1 + t, kv, valid)
```

Two ONNX Runtime calls per frame. Frames are 1/12.5 s of audio.

**Why `eoa_extra_frames` exists:** the control channel and the audio channels are
decoded from the *same* global hidden, so when `<eoa>` fires that frame's codes
already exist — and they are the model's own trailing silence. Dropping them cuts
the waveform the instant the last phone ends, clipping its release. `<eoa>` is
forbidden during the tail so those frames are real audio rather than an immediate
re-stop. The default is 1, which is what the published benchmark used.

**Classifier-free guidance:** at `cfg_scale > 1`, stack `[voice_emb; null_voice_emb]`
into a batch-2 sequence and pass `cfg_scale` to the decode graph, which
extrapolates the audio logits away from the unconditional row internally. The
sampled frame is tiled back across both rows — one frame is the history both
branches continue from. At `cfg_scale <= 1` the batch stays 1 and no
unconditional branch is computed at all.

## Codec

`onnx/codec/` holds the MOSS-Audio-Tokenizer-Nano **decoder** (Apache-2.0, see
[NOTICE](../NOTICE)). Two graphs:

* `decode_full` — `(B, T, K)` int32 codes → `(B, 2, T_audio)` float32, averaged
  to mono. One call for the whole utterance.
* `decode_step` — the same, KV-cached across chunks for streaming. Its state
  layout (per-decoder transformer offsets, per-layer key/value/position ring
  buffers) is described by `codec_browser_onnx_meta.json`'s `streaming_decode`
  section; feed each call's state outputs back in as the next call's inputs.

Two things bite ports here:

* **Every integer tensor in the codec is `int32`**, not int64. The TTS graphs use
  int64. Mixing them up fails at session-run time with a type error, which is at
  least loud.
* The codec's code axis order is **time-major, codebook-last** `(B, T, K)`, while
  the TTS loop produces `(B, K, T)`. One transpose, easy to forget.
* The position ring buffer initializes to **−1**, not 0 — position 0 is a real
  position, so a zero-filled buffer reads as "every slot holds frame 0".

The encoder graph (waveform → codes) is not shipped; see the README on voice
cloning.
