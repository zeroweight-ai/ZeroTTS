# Benchmarks

Full results behind the summary table in the [README](../README.md).

## Setup

| | |
|---|---|
| Benchmark | [`zeroweight-ai/ZeroBench-TTS`](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS) — 137 items, 59 held-out reference voices × 4 subsets |
| WER | [`vinai/PhoWhisper-large`](https://huggingface.co/vinai/PhoWhisper-large) — a Vietnamese-specialized Whisper finetune |
| SSIM | `microsoft/wavlm-base-plus-sv` x-vector cosine, generated vs. reference clip |
| UTMOS | UTMOSv2 naturalness MOS |
| Excess silence | leading + trailing + long mid-utterance pauses, in seconds |
| Coverage | 137/137 scored, 0 empty generations, for every model |

**Why PhoWhisper and not `whisper-large-v3`.** Raw Whisper is noticeably weaker on
Vietnamese, so scoring with it measures the ASR as much as the TTS system, and it
compresses the differences between models. Numbers here are therefore *not*
comparable to any published with `whisper-large-v3`.

**The two WER columns.** Every model is scored twice:

* **as-is** — WER is the minimum over the written `text` and the benchmark's
  hand-curated `text_normalized`, where one exists. Whisper is free to emit
  either "3,2%" or "ba phẩy hai phần trăm", and that choice is the ASR's, not the
  TTS model's; penalizing one would measure the ASR's formatting policy.
* **+vinorm** — adds a third accepted reference:
  [`soe-vinorm`](https://pypi.org/project/soe-vinorm/)'s automatic spoken-out
  normalization of the written text.

The `+vinorm` column exists to test a specific objection to this comparison —
that XTTS's stock tokenizer has no Vietnamese number/symbol expansion, so the
`challenging` gap reflects a missing frontend rather than the model. See
[Interpretation](#interpretation).

**ZeroTTS sampling** — the package defaults, so out-of-the-box output matches
these scores: `cfg_scale=1.0`, `audio_temperature=0.8`, `audio_topk=25`,
`audio_topp=0.95`, `audio_repetition_penalty=1.2`, `eoa_extra_frames=1`.

## Overall (n = 137)

| Model | WER as-is | WER +vinorm | WER median | SSIM | UTMOS | Excess silence (s) |
|---|---|---|---|---|---|---|
| **ZeroTTS** | **5.09%** | **4.51%** | **0.00%** | 0.936 | **2.95** | **0.029** |
| XTTS-v2-vietnamse | 21.49% | 21.16% | 5.56% | **0.940** | 2.36 | 0.532 |
| viXTTS | 25.13% | 24.92% | 13.51% | 0.935 | 2.35 | 0.233 |

## Per subset

## ZeroTTS breakdowns

By text length (as-is):

| Bucket | n | WER |
|---|---|---|
| short | 46 | 7.79% |
| medium | 46 | 3.24% |
| long | 45 | 4.21% |

Short texts score worst for every system in the comparison. A single misheard
word is a larger fraction of a short reference, so this is partly a property of
the metric rather than of the model.

By reference-voice source (as-is):

| Source | n | WER | SSIM |
|---|---|---|---|
| VIVOS | 57 | 3.65% | 0.929 |
| viVoice | 30 | 5.23% | 0.939 |
| phoaudiobook | 30 | 7.03% | 0.962 |
| Emilia (non-Vietnamese) | 20 | 6.05% | 0.911 |

## Repetition-penalty ablation

Why `audio_repetition_penalty` defaults to 1.2 (ZeroTTS, n=137):

| rp | WER as-is | WER +vinorm | SSIM | UTMOS | Excess silence (s) |
|---|---|---|---|---|---|
| **1.2** (default) | **5.09%** | **4.51%** | 0.936 | 2.95 | **0.029** |
| 1.0 | 5.90% | 5.33% | 0.936 | 2.94 | 0.040 |

About 0.8 pp of WER and a third less dead air, at no cost to voice similarity or
naturalness.

## Interpretation

**Intelligibility.** ZeroTTS leads on every subset, by 4–7× on `vietnamese`,
`cross_lingual` and `challenging`. Its median WER is 0.00% on three of four
subsets — the typical generation is transcribed exactly.

**The text-normalization objection, tested.** XTTS's stock tokenizer genuinely has
no Vietnamese number/symbol expansion. If that were the explanation for the
`challenging` gap, supplying the expansion on the scoring side should close it.
It does not: XTTS-v2-vietnamse moves 35.50% → 34.43%, viXTTS 41.68% → 40.93% —
roughly a point each, both still above 34% against ZeroTTS's 6.90%. The gap is
the model, not the frontend.

As a sanity check on the `+vinorm` reference itself: ZeroTTS gains where it
should (`challenging`, 8.86% → 6.90%) and is unchanged where there is nothing to
normalize (`vietnamese`, 0.16% → 0.16%). It is not simply loosening the metric
everywhere.

**Voice similarity is a tie, and one subset is a loss.** Overall SSIM is
0.936 / 0.940 / 0.935 — within noise. On `cross_lingual` ZeroTTS is genuinely
behind (0.911 vs ~0.935): it carries a foreign speaker's timbre into Vietnamese
less faithfully than the XTTS backbone, while still winning that subset's WER by
5×.

**Silence hygiene.** Both XTTS finetunes carry 0.21–0.61 s of excess silence per
clip against ZeroTTS's 0.014–0.037 s — lead-in and tail padding from the XTTS
decoder, which neither Vietnamese finetune addressed.

**Baseline caveat.** Both XTTS models ran through `coqui-tts` 0.27.5 (the
maintained fork of the abandoned `TTS` package) and needed two runtime patches to
work at all under a modern torch/transformers stack. Those patches are in the
eval harness and touch only the baselines.

## Reproducing

```bash
pip install "zerotts[eval]"

# generate + score, as-is
BENCHMARK=zeroweight-ai/ZeroBench-TTS \
MODEL=zerotts:zeroweight-ai/ZeroTTS \
OUT_DIR=./eval/zerotts_asis \
./evaluation/run_benchmark.sh

# re-score the SAME wavs with the vinorm reference — no re-synthesis
USE_VINORM=1 SKIP_EXISTING=1 \
BENCHMARK=zeroweight-ai/ZeroBench-TTS \
MODEL=zerotts:zeroweight-ai/ZeroTTS \
OUT_DIR=./eval/zerotts_vinorm \
./evaluation/run_benchmark.sh
```

Swap `MODEL=xtts:thivux/XTTS-v2-vietnamse` or `MODEL=xtts:capleaf/viXTTS` for the
baselines (needs `pip install coqui-tts`).

The `eval` extra installs PyTorch — the scorers are torch models. ZeroTTS
inference itself never needs it.
