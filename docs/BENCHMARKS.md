# Benchmarks

Full results for ZeroTTS on
**[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS)** —
137 items, 59 held-out reference voices × 4 subsets, against
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) and the two public
Vietnamese XTTS-v2 finetunes. Headline tables are in the
[README](../README.md#benchmarks); this page is the detail.

OmniVoice is given its optional `language="vi"` hint. Its model card recommends
it and it measurably helps — without it OmniVoice scores 5.15 % raw / 3.87 %
normalized instead of 4.13 % / 2.12 %, because the model otherwise infers the
language from the reference clip, which is wrong on `cross_lingual`.

## Speed — CPU

RTF (realtime factor, wall-clock synthesis time ÷ output audio duration —
lower is faster; below 1× is faster than real time) and time-to-first-audio,
all measured **on CPU**, single request, 8 inference threads pinned to a
dedicated core pool (no other synthesis running concurrently). Three
Vietnamese samples — short (26 chars), medium (77 chars), long (227 chars) —
each run 6 times with the first 2 (cold-cache) discarded; figures below are
the mean of the remaining 4.

| | **ZeroTTS** | OmniVoice | XTTS-v2-vietnamse | viXTTS |
|---|:-:|:-:|:-:|:-:|
| RTF — short | **0.51×** | 10.87× | 0.70× | 0.71× |
| RTF — medium | **0.47×** | 4.82× | 0.70× | 0.70× |
| RTF — long | **0.53×** | 2.67× | 0.71× | 0.78× |
| TTFA — short | **53 ms** | 21.7 s | 4.02 s | 2.45 s |
| TTFA — medium | **66 ms** | 28.9 s | 4.02 s | 3.72 s |
| TTFA — long | **89 ms** | 52.3 s | 10.3 s | 9.22 s |

ZeroTTS's time-to-first-audio comes from its real streaming path
(`synthesize_stream`/the `timing` arg to `synthesize`) — first audio frame,
not first full utterance. The three baselines have no working CPU streaming
path in this environment (OmniVoice's `generate()` returns the whole clip at
once; `coqui-tts`'s `inference_stream` throws against this stack's
`transformers` version — a compatibility break, not a deliberate limitation),
so their TTFA is the time to the complete utterance. That asymmetry is real
and worth naming, but it isn't the whole story: even OmniVoice's RTF —
generation time alone, no streaming involved — is 3-11× slower than real time
on CPU, because it's a GPU-sized model (3.1 GB vs. ZeroTTS's 0.86 GB) doing
autoregressive diffusion-LM decoding without CUDA kernels to lean on.

## How scoring works

**Nothing in this repo computes a metric.** `evaluation/run_benchmark.py`
synthesizes the 137 clips and hands them to `zerobench_eval`, the official
scorer published inside the benchmark dataset repo and fetched automatically by
`evaluation/zerobench.py`.

That split is the point: a scorer vendored into the repo of the model being
scored can drift from the benchmark's — silently, and always in the flattering
direction. The same code scores ZeroTTS and everything it is compared against,
and you can run it on any system without touching this repo.

The scorer's definition, in brief:

* **WER** — the minimum over **two ASRs**
  ([`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) and
  [`PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large)) and over
  **every acceptable reading** of the target text. Neither ASR can judge
  Vietnamese TTS alone: PhoWhisper cannot emit Latin script and re-spells
  embedded English phonetically ("Slack" → "sờ lếch"), while whisper-large-v3 is
  weaker on Vietnamese tone. References are expanded per surface span, so
  `31/12/2025` / "ba mốt tháng mười hai" / "31 tháng 12, 2025" all score 0 —
  but tone-only differences (`sảnh` → `sành`) stay errors, because Vietnamese
  tone is phonemic.
* **SSIM** — WavLM-base-plus-sv x-vector cosine vs. the reference clip.
* **UTMOS** — UTMOSv2 naturalness MOS (seeded, so it is reproducible).
* **Excess silence** — unwanted lead-in / tail / mid-utterance pause, in seconds.
  Nothing else catches dead air: an ASR happily transcribes a clip that opens
  with 1.5 s of nothing.

Full rationale, and the test suite that pins the reference policy in both
directions — format artifacts must score 0, real mispronunciations must still
cost — are in the
[benchmark README](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS).

## Headline — normalized text (n = 137)

Every system reads the benchmark's curated spoken-out form: dates, numbers and
acronyms already expanded. This is the condition a Vietnamese TTS system meets in production, where
a text frontend runs ahead of the model. ZeroTTS ships one
(`normalize_vi_text`, applied by default) which reproduces the benchmark's
reading on 34 of the 35 items that need normalization; neither baseline ships a
Vietnamese frontend at all.

| Model | **WER** | median | SSIM | UTMOS | Excess silence |
|---|---|---|---|---|---|
| **ZeroTTS** | **0.56 %** | **0.00 %** | 0.938 | **2.91** | **0.029 s** |
| OmniVoice | 2.12 % | 0.00 % | **0.951** | 2.75 | 0.386 s |
| XTTS-v2-vietnamse | 7.27 % | 0.00 % | 0.941 | 2.49 | 0.568 s |
| viXTTS | 8.61 % | 2.08 % | 0.935 | 2.34 | 0.215 s |

### Per subset — normalized text

| Subset | n | **ZeroTTS** | OmniVoice | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|---|
| `vietnamese` — monolingual | 39 | **0.21 %** | 0.50 % | 7.21 % | 7.54 % |
| `code_switch` — vi + English | 39 | 0.95 % | **0.46 %** | 10.14 % | 5.86 % |
| `cross_lingual` — foreign voice, vi text | 20 | **0.38 %** | 9.60 % | 4.94 % | 6.61 % |
| `challenging` — acronyms, dates, % | 39 | **0.61 %** | 1.56 % | 5.63 % | 13.44 % |
| **overall** | 137 | **0.56 %** | **2.12 %** | **7.27 %** | **8.61 %** |

### Per subset — raw text

The harder condition: the model is handed `31/12/2025` and `ChatGPT` verbatim
and has to read them itself, with no normalizer in front. This measures the
model's own grapheme-to-speech ability.

| Subset | n | **ZeroTTS** | OmniVoice | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|---|
| `vietnamese` — monolingual | 39 | **0.16 %** | 0.50 % | 7.92 % | 9.56 % |
| `code_switch` — vi + English | 39 | 0.97 % | **0.46 %** | 10.94 % | 9.25 % |
| `cross_lingual` — foreign voice, vi text | 20 | **1.42 %** | 17.71 % | 21.37 % | 27.27 % |
| `challenging` — acronyms, dates, % | 39 | **1.75 %** | 4.46 % | 27.86 % | 31.85 % |
| **overall** | 137 | **1.03 %** | **4.13 %** | **16.42 %** | **18.40 %** |

Raw-text overall — this is the table in the
[README](../README.md#benchmarks):

| Model | **WER** | median | SSIM | UTMOS | Excess silence |
|---|---|---|---|---|---|
| **ZeroTTS** | **1.03 %** | **0.00 %** | 0.936 | **2.91** | **0.029 s** |
| OmniVoice | 4.13 % | 0.00 % | **0.950** | 2.76 | 0.340 s |
| XTTS-v2-vietnamse | 16.42 % | 2.13 % | 0.940 | 2.43 | 0.532 s |
| viXTTS | 18.40 % | 6.38 % | 0.935 | 2.35 | 0.233 s |

### What normalization buys each system

| Model | raw text | normalized | change |
|---|---|---|---|
| **ZeroTTS** | **1.03 %** | **0.56 %** | −46 % |
| OmniVoice | 4.13 % | 2.12 % | −49 % |
| XTTS-v2-vietnamse | 16.42 % | 7.27 % | −56 % |
| viXTTS | 18.40 % | 8.61 % | −53 % |

`vietnamese` is the control: no digits, acronyms or English, so normalization
has nothing to do — and nothing moves (ZeroTTS 0.16 % → 0.21 %, one item of
sampling noise). The gains land exactly where the text needs expanding, which
is what makes the ablation meaningful rather than a free pass.

This is the strongest form of the objection "the baselines just lack a
Vietnamese text frontend", and it does not hold: they gain the most and still
lose by 13×, so the remaining gap is the acoustic model.

### Other metrics, per subset (raw-text runs)

| Subset | n | ZeroTTS SSIM / UTMOS / silence | OmniVoice | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|---|
| `vietnamese` | 39 | 0.936 / 2.84 / 0.027 s | 0.949 / 2.71 / 0.291 s | 0.935 / 2.46 / 0.580 s | 0.933 / 2.30 / 0.218 s |
| `code_switch` | 39 | 0.945 / 2.95 / 0.030 s | 0.955 / 2.76 / 0.350 s | 0.946 / 2.44 / 0.448 s | 0.939 / 2.42 / 0.221 s |
| `cross_lingual` | 20 | 0.911 / 3.02 / 0.014 s | 0.948 / 2.90 / 0.274 s | 0.936 / 2.38 / 0.457 s | 0.935 / 2.45 / 0.207 s |
| `challenging` | 39 | 0.941 / 2.90 / 0.037 s | 0.949 / 2.74 / 0.412 s | 0.939 / 2.43 / 0.606 s | 0.933 / 2.28 / 0.272 s |

## Interpretation

* **Voice similarity is a tie, not a win.** 0.938 / 0.941 / 0.935 normalized
  (0.936 / 0.940 / 0.935 raw) is within noise either way. On `cross_lingual`
  ZeroTTS is behind on the raw runs (0.911 vs ~0.935): it carries a foreign
  speaker's timbre into Vietnamese slightly less faithfully, while winning that
  subset's WER by 15×.
* **Both XTTS finetunes carry far more dead air** (0.22–0.57 s vs 0.029 s) —
  lead-in/tail padding from XTTS's decoder, in both input modes.
* **Reproducibility.** ZeroTTS inference is seeded; re-running the whole
  benchmark from scratch reproduced these WER figures exactly.

## Reproducing

```bash
pip install "zerotts[eval]"

# synthesize + score (the scorer is fetched from the benchmark repo)
./evaluation/run_benchmark.sh

# or explicitly
python evaluation/run_benchmark.py \
    --benchmark zeroweight-ai/ZeroBench-TTS \
    --model zerotts:zeroweight-ai/ZeroTTS \
    --out_dir ./eval/zerotts

# the text-normalization ablation
SYNTH_FROM=text_normalized OUT_DIR=./eval/zerotts_norm ./evaluation/run_benchmark.sh

# baselines (needs `pip install coqui-tts`)
MODEL=xtts:thivux/XTTS-v2-vietnamse OUT_DIR=./eval/xtts_vi ./evaluation/run_benchmark.sh
MODEL=xtts:capleaf/viXTTS           OUT_DIR=./eval/vixtts  ./evaluation/run_benchmark.sh
```

Already have wavs from some other system? Skip this repo entirely:

```bash
huggingface-cli download zeroweight-ai/ZeroBench-TTS --repo-type dataset --local-dir ZeroBench-TTS
cd ZeroBench-TTS && pip install -r zerobench_eval/requirements.txt
python -m zerobench_eval manifest --out manifest.jsonl    # what to synthesize
python -m zerobench_eval score --wav_dir my_wavs/ --name MyModel
```
