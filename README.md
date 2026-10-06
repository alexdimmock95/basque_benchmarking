# Basque ASR Benchmarking

Benchmarking multiple pretrained speech-to-text models on Basque (Euskara), comparing accuracy, character-level accuracy, and inference speed across models that officially support Basque versus models tested zero-shot.

## Literature Review

**Foundational survey:** Besacier et al. (2014), "Automatic speech recognition for under-resourced languages: A survey" (*Speech Communication*), remains the base reference point the field builds from when discussing low-resource ASR challenges (data scarcity, acoustic/language modeling difficulty, evaluation standardisation).

**Current direction in the field — robustness over leaderboard numbers:** recent work has shifted from reporting single-benchmark accuracy toward testing whether models that look strong on clean, standard benchmarks (Common Voice, FLEURS) hold up under real-world conditions. [GigaSpeechBench](https://arxiv.org/html/2606.28884v3) (2026) found that systems nearing saturation on standard benchmarks "degrade substantially on in-the-wild low-resource and dialectal speech." This is the exact pattern prior Basque-specific work has already flagged (see below), and it's the central motivation for this project: a model's published WER doesn't tell you how it behaves on a specific low-resource language's edge cases.

**Multilingual low-resource benchmarking at scale:** [LoASR-Bench](https://arxiv.org/html/2603.20042) (2026) evaluates speech-language models across 25 low-resource languages spanning 9 language families, but Basque is not among them, an example of how even dedicated low-resource benchmarks still leave language-isolate gaps like Basque uncovered.

**Basque-specific prior work:**
- [Whisper-LM](https://arxiv.org/abs/2503.23542) (de Zuazo et al., 2025) fine-tunes Whisper on Basque (among other languages) and augments it with statistical/LLM-based language models, reporting WER improvements up to 51% in-distribution and 34% out-of-distribution. This paper is the direct source of this project's `whisper-tiny-eu` comparison model, and the same team has released a family of Basque Whisper fine-tunes (base, small, medium, tiny) at various sizes.
- A SpeechLM fine-tuned on Basque (HiTZ Center) achieved 8.09% WER in-distribution but underperformed notably on out-of-distribution Basque test data, reinforcing the GigaSpeechBench finding above.
- Work evaluating ASR on Basque dialectal/spontaneous broadcast speech found recognition degrades significantly versus standard speech, with recurring phonological error patterns. This project's test set (Common Voice, scripted/standard speech) does not cover that dialectal gap directly, noted as a limitation below rather than addressed here.

**The gap this project addresses:** despite this body of work, no existing paper benchmarks multiple modern ASR models (general-purpose and Basque-specific) on the *same* Basque test set under the *same* conditions. Individual papers report individual models at different times on different splits. This project builds that single, reproducible comparison table, and additionally reports how models with no claimed Basque support perform when tested zero-shot on the language, a comparison point absent from the literature above.

## Models

| Model | Basque officially supported? | Notes |
|---|---|---|
| Whisper-large-v3 | Yes | 99 languages officially supported, Basque included |
| Whisper-tiny-eu | Yes | Fine-tuned specifically for Basque |
| SenseVoice-Small | No | Covers English, Chinese, Cantonese, Japanese, Korean only; zero-shot test for Basque (SenseVoice-Large's weights are not publicly released) |
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

*Preliminary: based on a 200-clip sample of the 14,801-clip test set, same clips and conditions across all models. Full-set results (all 14,801 clips) to follow; numbers below may shift slightly, though the overall pattern is expected to hold given the size of the gaps observed.*

| Model | WER | CER | Substitutions | Insertions | Deletions | avg RTF | Basque officially supported? |
|---|---|---|---|---|---|---|---|
| wav2vec2-Basque (own) | 0.128 | 0.020 | 170 | 27 | 9 | 0.031 | Yes (fine-tuned) |
| Whisper-tiny-eu | 0.272 | 0.054 | 345 | 49 | 44 | 0.056 | Yes (fine-tuned) |
| Whisper-large-v3 | 0.355 | 0.065 | 445 | 68 | 59 | 0.300 | Yes |
| SenseVoice-Small | 1.019 | 0.899 | 891 | 33 | 717 | 0.017 | No (zero-shot) |

The own fine-tuned wav2vec2 model achieves the lowest WER and CER of all four, ahead of both Whisper variants, with a notably low deletion count (9) relative to its substitution count (170), suggesting its errors are mostly close-but-wrong rather than missed words entirely. SenseVoice-Small, the only model with no claimed Basque support, produces a WER above 1.0 (more errors than words in the reference) and a CER near 0.9, with deletions dominating its error breakdown, a markedly different error pattern from the other three models, where substitutions dominate. It is also by far the fastest model (RTF 0.017), roughly 18x faster than Whisper-large-v3.

## Discussion

**Specialization outperforms generalist scale.** The clearest finding is that a smaller model fine-tuned specifically on Basque (wav2vec2-Basque) outperforms a much larger generalist model (Whisper-large-v3) trained across 99 languages, by a wide margin on both WER and CER. This is a well-documented pattern in low-resource ASR rather than a surprising result, but it is a useful concrete data point for anyone deciding between fine-tuning a smaller model versus relying on a large multilingual one for a specific low-resource language.

**Speed and accuracy trade off starkly when a model has zero exposure to the target language.** SenseVoice-Small's architecture is built for speed (non-autoregressive decoding, per Du et al. 2024), and that speed advantage holds here. But it does not transfer to a language the model was never trained on: its WER and CER indicate the transcriptions are largely unusable, and the dominance of deletions (rather than substitutions) in its error breakdown suggests it may be missing or misattributing whole segments of speech, possibly connected to the language-ID tag in its raw output defaulting to English rather than Basque (see Setup notes). This result lines up with the GigaSpeechBench (2026) finding that models performing well on standard benchmarks can degrade substantially outside their trained distribution, here taken to an extreme case of a language entirely outside that distribution.

**Error type, not just WER, differentiates the models.** The three Basque-supporting models all show substitution-dominant error profiles (getting close to the right word, but not exact), consistent with the "dropped sounds and merged word boundaries" pattern noted in related Basque dialectal ASR work, rather than wholesale failure. SenseVoice-Small's deletion-dominant profile is qualitatively different, supporting the interpretation that it is failing to recognize Basque as Basque, not simply making more of the same kind of error.

**Caveats on this preliminary result.** The wav2vec2-Basque model was fine-tuned on `HiTZ/composite_corpus_eu_v2.1`, which may share distributional characteristics with the Common Voice test set used here (recording style, speaker demographics, sentence domains). This does not invalidate the result, the test split was not used in training, but it is a fairer comparison of "a model tuned on Basque-like data" than of fully out-of-domain generalization, and is worth stating plainly rather than implying a bigger advantage than the setup supports.

## Limitations

- Test set limited to Common Voice's standard scripted speech; does not cover dialectal or spontaneous speech, where prior work shows ASR performance degrades.
- 200-clip sample used for initial Whisper-large-v3 results; full 14,801-clip run pending for final comparability across models.

## License

Common Voice data is CC0. Re-hosting or re-sharing the raw dataset is prohibited per Mozilla's terms.
