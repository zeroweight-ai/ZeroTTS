<div align="center">

# ZeroTTS

**Vietnamese text-to-speech · zero PyTorch · CPU-first · streaming**

[![PyPI](https://img.shields.io/pypi/v/zerotts?color=3775AB)](https://pypi.org/project/zerotts/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/zeroweight-ai/ZeroTTS)

</div>

ZeroTTS synthesizes Vietnamese speech from text. The entire inference path is
**numpy + ONNX Runtime** — no PyTorch, no CUDA, no build step — so it runs on a
laptop CPU, in a container, or (see [`js/`](js/)) in a browser.

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
audio = tts.synthesize("Xin chào các bạn, mình là ZeroTTS.", voice="arya")
tts.save_audio(audio, "out.wav")
```

## Contents

1. [Install](#install)
2. [Usage](#usage)
3. [Voices — and voice cloning](#voices--and-voice-cloning)
4. [Benchmarks](#benchmarks)
5. [Web UI](#web-ui)
6. [Browser demo](#browser-demo)
7. [How it works](#how-it-works)
8. [Credits](#credits)

---

## Install

```bash
pip install zerotts
```

That pulls `numpy`, `onnxruntime`, `tokenizers`, `huggingface_hub`, `soundfile`,
`scipy` — and nothing else. Weights download from the Hub on first use
(~900 MB, cached under `HF_HOME`).

Optional extras:

```bash
pip install "zerotts[webui]"   # Gradio demo
pip install "zerotts[eval]"    # benchmark scorers — these DO need torch
```

The `eval` extra is the only thing in this repo that installs PyTorch, and it is
for *measuring* quality, not for generating audio.

## Usage

### Python

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

print(tts.list_voices())

# One shot
audio = tts.synthesize("Hôm nay trời đẹp quá.", voice="arya")
tts.save_audio(audio, "out.wav")

# Streaming — first chunk arrives in ~100 ms, then chunks ramp up in size
for chunk in tts.synthesize_stream("Một đoạn văn bản dài hơn…", voice="arya"):
    play(chunk)   # (1, n) float32 at 48 kHz
```

Dates, clock times, fractions and acronyms are expanded to spoken Vietnamese
before synthesis:

```python
from zerotts import normalize_vi_text

normalize_vi_text("Ngày 23/8/2024 lúc 15h30, giá 1.250.000")
# 'Ngày hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư lúc
#  mười lăm giờ ba mươi phút, giá một triệu hai trăm năm mươi nghìn'
```

`synthesize()` does **not** apply it — it is a separate step so you stay in
control (the expansions are Vietnamese words, so they are wrong for English
text). The CLI and web UI apply it by default; `zerotts say --no_text_norm`
turns it off.

Long input should be segmented — the model is trained on utterances, not
paragraphs:

```python
from zerotts.chunking import chunk_text, clean_segment_punctuation, normalize_punctuation

segments = [clean_segment_punctuation(s)
            for s in chunk_text(normalize_punctuation(long_text), max_chunk_sec=15)]
```

### Command line

```bash
zerotts voices
zerotts say "Xin chào các bạn." --voice arya -o hello.wav
zerotts say "$(cat article.txt)" --voice arya --chunk -o article.wav
zerotts bench --voice arya
```

### Generation settings

| Argument | Default | Effect |
|---|---|---|
| `voice` | `None` | Voice pack name. `None` = the model's unconditional voice, which is *not* stable across runs. |
| `cfg_scale` | `1.0` | `>1` guides toward the voice's identity, at 2× the per-frame cost. |
| `audio_temperature` | `0.8` | |
| `audio_topk` / `audio_topp` | `25` / `0.95` | |
| `audio_repetition_penalty` | `1.2` | Benchmarked default. `1.0` measurably raises WER (5.90% vs 5.09%) and leaves more dead air. |
| `eoa_extra_frames` | `1` | Frames of trailing audio kept after the model signals stop. `0` clips the last phone's release. |

Defaults are the exact settings the [benchmark numbers](#benchmarks) were
produced with, so out-of-the-box output matches the published scores.

## Voices — and voice cloning

A voice in ZeroTTS is a small array of speaker latents, shape
`(1, n_voice_queries, d_model)`. That array is the *entire* speaker
conditioning — there is no reference transcript, no in-context audio prompt, no
teacher-forced frames. It ships as a `.npz` inside the weights repo.

> ### Voice cloning is not available in this release
>
> Those latents are produced by a voice encoder that reads a reference clip, and
> **the voice encoder is not published**. This package can load voices; it cannot
> create them from audio. There is no flag that turns this on.
>
> To get latents for your own speaker, visit
> **[zeroweight.ai](https://zeroweight.ai)** or get in touch.

The boundary is narrower than it sounds: latents obtained that way are just a
`.npz`, so they drop into `voices/<name>/voice.npz` and work with no code change.

```python
tts.list_voices()                       # ['arya', ...]
v = tts.load_voice("arya")
v.emb.shape                             # (1, 10, 768)

# A latent array from anywhere works directly
audio = tts.synthesize("…", voice=my_latents)
```

## Benchmarks

Measured on **[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS)** —
137 items, 59 held-out reference voices × 4 subsets — against the two public
Vietnamese XTTS finetunes. 137/137 scored, 0 empty generations.

* **WER** — [`vinai/PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large),
  a Vietnamese-specialized Whisper finetune. Scoring Vietnamese TTS with raw
  `whisper-large-v3` measures the ASR's weakness as much as the TTS system's.
* **SSIM** — WavLM-base-plus-sv x-vector cosine vs. the reference clip.
* **UTMOS** — UTMOSv2 naturalness MOS. **Silence** — excess lead-in/tail/mid-utterance silence.
* Every model is reported **twice**: `as-is`, and `+vinorm` which adds
  [`soe-vinorm`](https://pypi.org/project/soe-vinorm/)'s automatic spoken-out
  normalization as an extra accepted reference. See the note below.

### Overall (n = 137)

| Model | WER as-is | WER +vinorm | SSIM | UTMOS | Excess silence |
|---|---|---|---|---|---|
| **ZeroTTS** | **5.09 %** | **4.51 %** | 0.936 | **2.95** | **0.029 s** |
| XTTS-v2-vietnamse | 21.49 % | 21.16 % | **0.940** | 2.36 | 0.532 s |
| viXTTS | 25.13 % | 24.92 % | 0.935 | 2.35 | 0.233 s |

### Per subset (WER, as-is → +vinorm)

| Subset | n | ZeroTTS | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|
| `vietnamese` — monolingual | 39 | **0.16 % → 0.16 %** | 8.18 % → 8.18 % | 11.25 % → 11.25 % |
| `code_switch` — vi + English | 39 | **5.74 % → 5.72 %** | 16.16 % → 16.15 % | 17.66 % → 17.64 % |
| `cross_lingual` — foreign voice, vi text | 20 | **6.05 % → 5.98 %** | 30.49 % → 30.37 % | 34.51 % → 34.51 % |
| `challenging` — acronyms, dates, % | 39 | **8.86 % → 6.90 %** | 35.50 % → 34.43 % | 41.68 % → 40.93 % |

Full tables (SSIM/UTMOS/silence per subset, length buckets, voice sources, the
repetition-penalty ablation) and reproduction commands are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Reading these fairly:**

* **The `+vinorm` column exists to test an objection, not to flatter us.** XTTS's
  stock tokenizer has no Vietnamese number/symbol expansion, so one could argue
  the `challenging` gap is a missing text-normalization frontend rather than the
  model. Supplying that normalization on the scoring side moves XTTS
  35.50 % → 34.43 % and viXTTS 41.68 % → 40.93 % — about a point each, both still
  above 34 %. It does not close a 26-point gap.
* **Voice similarity is a tie, not a win.** 0.936 / 0.940 / 0.935 is within noise.
  On `cross_lingual` specifically ZeroTTS is behind (0.911 vs ~0.935): it carries
  a foreign speaker's timbre into Vietnamese slightly less faithfully.
* `challenging` is the hardest subset for every system, and short texts score
  worse than long ones for all of them (one misheard word is a bigger fraction of
  a short reference).

## Web UI

```bash
pip install "zerotts[webui]"
python webui/app.py                       # http://localhost:7860
python webui/app.py --model ./local_dir   # a local model directory
```

Voice picker, streaming playback, long-form segmentation, and the generation
settings above.

## Browser demo

[`js/`](js/) runs the same model client-side with `onnxruntime-web` — no server,
no upload. See [docs/BROWSER.md](docs/BROWSER.md).

Note the download: the weights are **fp32 and not quantized**, so the demo fetches
~900 MB once and persists it (OPFS/Cache API). That is a deliberate
quality-over-size choice; it targets desktop broadband, not mobile data.

## How it works

Three ONNX graphs and a two-level autoregressive loop:

| Graph | Runs | Does |
|---|---|---|
| `text_encoder.onnx` | once per utterance | text ids → encoder states |
| `prefix_step.onnx` | once + **once per frame** | advances the global (time) transformer, KV-cached |
| `local_frame_decode.onnx` | **once per frame** | decodes one whole frame: stop-or-continue + all 16 codebooks, with embeddings, CFG mixing and sampling fused in |

Two ONNX Runtime calls per audio frame. Frames come out at 12.5 Hz and are
turned into a 48 kHz waveform by the bundled MOSS codec decoder, either in one
batch or through a KV-cached streaming decoder.

Sampling (temperature / top-k / top-p / repetition penalty) and the stop decision
live *inside* `local_frame_decode.onnx`, not in Python — which is what lets the
identical pipeline run under `onnxruntime-web`, where there is no Python at all.

More detail — tensor shapes, the KV-cache contract, the position-id convention —
in [docs/RUNTIME.md](docs/RUNTIME.md).

**Not included:** model architecture, training code, and the ONNX export script
are not part of this repository, and the voice encoder is not published (see
[Voices](#voices--and-voice-cloning)).

## Credits

Speech codec: **[MOSS-Audio-Tokenizer-Nano](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)**
by the OpenMOSS team, Apache-2.0. ZeroTTS bundles its ONNX **decoder** graphs in
the weights repo so there is no external runtime dependency; see
[NOTICE](NOTICE) and [LICENSES/](LICENSES/).

```bibtex
@misc{gong2026mossaudiotokenizerscalingaudiotokenizers,
  title={MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models},
  author={Yitian Gong and Kuangwei Chen and Zhaoye Fei and Xiaogui Yang and Ke Chen
          and Yang Wang and Kexin Huang and Mingshu Chen and Ruixiao Li
          and Qingyuan Cheng and Shimin Li and Xipeng Qiu},
  year={2026}, eprint={2602.10934}, archivePrefix={arXiv}, primaryClass={cs.SD}
}
```

Vietnamese text normalization adapts the expansion rules and abbreviation table
of **[soe-vinorm](https://github.com/vinhdq842/soe-vinorm)** (MIT), reimplemented
as pure stdlib regex so the inference path keeps its no-torch, no-download
guarantee. See [NOTICE](NOTICE).

Benchmark reference audio comes from
[VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos),
[viVoice](https://huggingface.co/datasets/capleaf/viVoice),
[phoaudiobook](https://huggingface.co/datasets/thivux/phoaudiobook) and
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset).
ASR scoring uses [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) (VinAI).

## License

Code and weights: **MIT**. The bundled MOSS codec decoder is Apache-2.0.
The ZeroBench-TTS *dataset* is CC-BY-NC-4.0 (it redistributes audio from the
corpora above) — that applies to the benchmark, not to ZeroTTS.
