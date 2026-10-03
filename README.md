# Basque ASR Benchmarking

Benchmarking multiple pretrained speech-to-text models on Basque (Euskara), comparing accuracy, character-level accuracy, and inference speed across models that officially support Basque versus models tested zero-shot.

## Literature Review
 
**Foundational survey:** Besacier et al. (2014), "Automatic speech recognition for under-resourced languages: A survey" (*Speech Communication*), remains the base reference point the field builds from when discussing low-resource ASR challenges (data scarcity, acoustic/language modeling difficulty, evaluation standardisation).
 
**Current direction in the field — robustness over leaderboard numbers:** recent work has shifted from reporting single-benchmark accuracy toward testing whether models that look strong on clean, standard benchmarks (Common Voice, FLEURS) hold up under real-world conditions. [GigaSpeechBench](https://arxiv.org/html/2606.28884v3) (2026) found that systems nearing saturation on standard benchmarks "degrade substantially on in-the-wild low-resource and dialectal speech." This is the exact pattern prior Basque-specific work has already flagged (see below), and it's the central motivation for this project: a model's published WER doesn't tell you how it behaves on a specific low-resource language's edge cases.
 
**Multilingual low-resource benchmarking at scale:** [LoASR-Bench](https://arxiv.org/html/2603.20042) (2026) evaluates speech-language models across 25 low-resource languages spanning 9 language families, but Basque is not among them, an example of how even dedicated low-resource benchmarks still leave language-isolate gaps like Basque uncovered.
 
**Basque-specific prior work:**
- [Whisper-LM](https://arxiv.org/abs/2503.23542) (de Zuazo et al., 2025) fine-tunes Whisper on Basque (among other languages) and augments it with statistical/LLM-based language models, reporting WER improvements up to 51% in-distribution and 34% out-of-distribution. This paper is the direct source of this project's `whisper-tiny-eu` comparison model, and the same team has released a family of Basque Whisper fine-tunes (base, small, medium, tiny) at various sizes.
- A SpeechLM fine-tuned on Basque (HiTZ Center) achieved 8.09% WER in-distribution but underperformed notably on out-of-distribution Basque test data, reinforcing the GigaSpeechBench finding above.
- Work evaluating ASR on Basque dialectal/spontaneous broadcast speech found recognition degrades significantly versus standard speech, with recurring phonological error patterns, directly motivating this project's accent-based error breakdown.
**The gap this project addresses:** despite this body of work, no existing paper benchmarks multiple modern ASR models (general-purpose and Basque-specific) on the *same* Basque test set under the *same* conditions. Individual papers report individual models at different times on different splits. This project builds that single, reproducible comparison table, and additionally reports how models with no claimed Basque support perform when tested zero-shot on the language, a comparison point absent from the literature above.

## Models

| Model | Basque officially supported? | Notes |
|---|---|---|
| Whisper-large-v3 | Yes | 99 languages officially supported, Basque included |
| Whisper-tiny-eu | Yes | Fine-tuned specifically for Basque |
| SenseVoice-Large | No | Covers English, Chinese, Cantonese, Japanese, Korean only; zero-shot test for Basque |
| wav2vec2-Basque (this project) | Yes | Fine-tuned specifically for Basque |

## Data

[Common Voice Basque (v27)](https://datacollective.mozillafoundation.org), official test split: 14,801 clips. Downloaded via the Mozilla Data Collective API (Common Voice is no longer hosted on HuggingFace as of October 2025).

## Setup

```bash
pip install -q jiwer
pip install -q -U transformers datasets accelerate
```

Requires a HuggingFace token (`HF_TOKEN`) and a Mozilla Data Collective API key, stored as Kaggle secrets.

## Metrics

- **WER** (Word Error Rate) — word-level accuracy
- **CER** (Character Error Rate) — character-level accuracy, important for Basque given its long, agglutinative word forms
- **RTF** (Real-Time Factor) — inference time relative to audio duration; lower is faster
- **Error breakdown** — substitutions, insertions, deletions per model

## Results

| Model | WER | CER | RTF (avg) |
|---|---|---|---|
| Whisper-large-v3 | 0.355 | 0.065 | TBD |
| Whisper-tiny-eu | TBD | TBD | TBD |
| SenseVoice-Large | TBD | TBD | TBD |
| wav2vec2-Basque | TBD | TBD | TBD |

*(Whisper-large-v3 results from a 200-clip sample of the test set; full-set results to follow.)*

## Limitations

- Test set limited to Common Voice's standard scripted speech; does not cover dialectal or spontaneous speech, where prior work shows ASR performance degrades.
- 200-clip sample used for initial Whisper-large-v3 results; full 14,801-clip run pending for final comparability across models.

## License

Common Voice data is CC0. Re-hosting or re-sharing the raw dataset is prohibited per Mozilla's terms.
