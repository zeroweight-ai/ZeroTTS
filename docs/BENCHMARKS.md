# Benchmarks

Full results for ZeroTTS on
**[ZeroBench-TTS](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS)** —
137 items, 59 held-out reference voices × 4 subsets. Headline tables are in the
[README](../README.md#benchmarks); this page is the detail.

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

Three reference policies are reported on every run, so you can see how much of a
number is scoring policy rather than synthesis:

| policy | references |
|---|---|
| `strict` | the written text, verbatim |
| `norm` | + the benchmark's curated spoken-out form |
| `robust` | + every acceptable reading — **the headline** |

Full rationale, and the test suite that pins the policy in both directions, are
in the [benchmark README](https://huggingface.co/datasets/zeroweight-ai/ZeroBench-TTS).

## Overall (n = 137)

| Model | WER strict | WER norm | **WER robust** | median | SSIM | UTMOS | Excess silence |
|---|---|---|---|---|---|---|---|
| **ZeroTTS** | **5.26 %** | **2.96 %** | **1.03 %** | **0.00 %** | 0.936 | **2.91** | **0.029 s** |
| XTTS-v2-vietnamse | 18.83 % | 17.82 % | 16.42 % | 2.13 % | **0.940** | 2.43 | 0.532 s |
| viXTTS | 20.22 % | 19.47 % | 18.40 % | 6.38 % | 0.935 | 2.35 | 0.233 s |

Note the asymmetry between policies: ZeroTTS drops 5.1× from `strict` to
`robust`, the baselines only 1.15×. Most of ZeroTTS's residual was formatting;
theirs is hallucinated and garbled speech, which no reference policy can excuse.

## Per subset

WER, `robust` policy:

| Subset | n | ZeroTTS | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|
| `vietnamese` — monolingual | 39 | **0.16 %** | 7.92 % | 9.56 % |
| `code_switch` — vi + English | 39 | **0.97 %** | 10.94 % | 9.25 % |
| `cross_lingual` — foreign voice, vi text | 20 | **1.42 %** | 21.37 % | 27.27 % |
| `challenging` — acronyms, dates, % | 39 | **1.75 %** | 27.86 % | 31.85 % |

Other metrics:

| Subset | n | ZeroTTS SSIM / UTMOS / silence | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|---|
| `vietnamese` | 39 | 0.936 / 2.84 / 0.027 s | 0.935 / 2.46 / 0.580 s | 0.933 / 2.30 / 0.218 s |
| `code_switch` | 39 | 0.945 / 2.95 / 0.030 s | 0.946 / 2.44 / 0.448 s | 0.939 / 2.42 / 0.221 s |
| `cross_lingual` | 20 | 0.911 / 3.02 / 0.014 s | 0.936 / 2.38 / 0.457 s | 0.935 / 2.45 / 0.207 s |
| `challenging` | 39 | 0.941 / 2.90 / 0.037 s | 0.939 / 2.43 / 0.606 s | 0.933 / 2.28 / 0.272 s |

## Text-normalization ablation

Feed every model the spoken-out `text_normalized` instead of raw orthography
(`SYNTH_FROM=text_normalized`). This simulates a perfect Vietnamese
text-normalization frontend and separates grapheme-to-spoken-form errors from
acoustic ones. Scoring references are unchanged, so the two columns are
directly comparable.

| Model | raw text | pre-normalized | change |
|---|---|---|---|
| **ZeroTTS** | **1.03 %** | **0.56 %** | −46 % |
| XTTS-v2-vietnamse | 16.42 % | 7.27 % | −56 % |
| viXTTS | 18.40 % | 8.61 % | −53 % |

Per subset, `robust` WER, raw → pre-normalized:

| Subset | ZeroTTS | XTTS-v2-vietnamse | viXTTS |
|---|---|---|---|
| `vietnamese` | 0.16 % → 0.21 % | 7.92 % → 7.21 % | 9.56 % → 7.54 % |
| `code_switch` | 0.97 % → 0.95 % | 10.94 % → 10.14 % | 9.25 % → 5.86 % |
| `cross_lingual` | 1.42 % → 0.38 % | 21.37 % → 4.94 % | 27.27 % → 6.61 % |
| `challenging` | 1.75 % → 0.61 % | 27.86 % → 5.63 % | 31.85 % → 13.44 % |

`vietnamese` is the control — no digits, acronyms or English, so normalization
has nothing to do, and nothing moves. The gains land exactly where the text
needs expanding.

This is the strongest form of the objection "the baselines just lack a
Vietnamese text frontend", and it does not hold: they gain the most and still
lose by 13–15×, so the remaining gap is the acoustic model.

## ZeroTTS's remaining errors

110 of 137 items are transcribed exactly; median WER is 0.00 % on all four
subsets and the worst single item is 0.143. Every item above 0.00 was audited by
transcribing it with both ASRs and asking whether they agree:

| Verdict | Items |
|---|---|
| Real synthesis defect (both ASRs converge on the same wrong output) | 20 |
| ASR spelling disagreement (audio intelligible, no two transcripts agree) | 7 |
| Benchmark artifact | 0 |

Three failure families account for nearly all of it:

1. **Voiced leading zeros in dates** — `18/04` → "tháng **không** tư",
   `01/07` → "ngày **không** một". Five items across four voices, so it is the
   text frontend, not sampling. The largest remaining family.
2. **Latin acronyms containing W or H** — `WHO` → "Hall"/"Hồ"/"BTHO",
   `WTO` → "NETW". `VN`, `GDP`, `UNESCO`, `UNICEF`, `ASEAN` are all fine; the
   failure is specific to the Vietnamese letter names for `W` and `H`.
3. **Tone errors on low-frequency syllables** — `sảnh` → `sành`,
   `bỏ dở` → `bỏ giờ`, `nhàu` → `nhau`. The part a native listener notices
   first, and the only family that is genuinely acoustic.

Families 1 and 2 are grapheme-to-spoken-form bugs, which is why the ablation
above halves the total. Notably absent: no hallucinated tails, loops, drift, or
truncation — the failure modes that dominate both XTTS baselines' worst cases.

## Interpretation

* **Voice similarity is a tie, not a win.** 0.936 / 0.940 / 0.935 is within
  noise. On `cross_lingual` ZeroTTS is genuinely behind (0.911 vs ~0.935): it
  carries a foreign speaker's timbre into Vietnamese slightly less faithfully,
  while winning that subset's WER by 15×.
* **Both XTTS finetunes carry far more dead air** (0.23–0.53 s vs 0.029 s) —
  lead-in/tail padding from XTTS's decoder.
* **Reproducibility.** ZeroTTS inference is seeded; re-running the whole
  benchmark from scratch reproduced these WER figures exactly at every policy.

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
