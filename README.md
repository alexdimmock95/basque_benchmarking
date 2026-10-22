# Basque ASR Benchmarking

Benchmarking multiple pretrained speech-to-text models on Basque (Euskara), comparing accuracy, character-level accuracy, and inference speed across models that officially support Basque versus models tested zero-shot.

## Literature Review

**Foundational survey:** Besacier et al. (2014), "Automatic speech recognition for under-resourced languages: A survey" (*Speech Communication*), remains the base reference point the field builds from when discussing low-resource ASR challenges (data scarcity, acoustic/language modeling difficulty, evaluation standardisation).

**Current direction in the field: robustness over leaderboard numbers:** recent work has shifted from reporting single-benchmark accuracy toward testing whether models that look strong on clean, standard benchmarks (Common Voice, FLEURS) hold up under real-world conditions. [GigaSpeechBench](https://arxiv.org/html/2606.28884v3) (2026) found that systems nearing saturation on standard benchmarks "degrade substantially on in-the-wild low-resource and dialectal speech." This is the exact pattern prior Basque-specific work has already flagged (see below), and it's the central motivation for this project: a model's published WER doesn't tell you how it behaves on a specific low-resource language's edge cases.

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

- **WER** (Word Error Rate): word-level accuracy
- **CER** (Character Error Rate): character-level accuracy, important for Basque given its long, agglutinative word forms
- **RTF** (Real-Time Factor): inference time relative to audio duration; lower is faster
- **Error breakdown**: substitutions, insertions, deletions per model

## Results

Full official test split of Common Voice Basque v27 (14,801 clips), same clips and conditions across all four models.

| Model | WER | CER | Substitutions | Insertions | Deletions | avg RTF | Basque officially supported? |
|---|---|---|---|---|---|---|---|
| wav2vec2-Basque (own) | 0.155 | 0.028 | 14,929 | 1,927 | 1,411 | 0.031 | Yes (fine-tuned) |
| Whisper-tiny-eu | 0.332 | 0.081 | 30,529 | 4,608 | 3,855 | 0.058 | Yes (fine-tuned) |
| Whisper-large-v3 | 0.451 | 0.111 | 38,368 | 9,485 | 5,115 | 0.344 | Yes |
| SenseVoice-Small | 1.015 | 0.890 | 60,118 | 1,946 | 57,263 | 0.018 | No (zero-shot) |

The own fine-tuned wav2vec2 model has the lowest WER and CER of the four, ahead of both Whisper variants. SenseVoice-Small, the only model with no claimed Basque support, has a WER above 1.0 (more errors than words in the reference) and a CER near 0.9. It is also the fastest model (RTF 0.018), roughly 20x faster than Whisper-large-v3.

I first ran 200 and then 2,000 clips as sanity checks. The ranking was identical at all three sizes, but WER crept up for every model except SenseVoice as the sample grew (for example wav2vec2-Basque: 0.128, 0.142, 0.155). The first rows of the test file seem to be easier than the rest, so only the full-set numbers are reported here.

## Train/test overlap check

The wav2vec2-Basque model was trained on the first 128,000 rows of `HiTZ/composite_corpus_eu_v2.1`, which includes Common Voice material, so I checked whether test sentences also appear in those rows. The corpus has no clip IDs, so I matched on text (lowercased, punctuation removed, exact match). This shows the same sentence appears in both, not the same audio, since a different speaker may have read it.

1,739 test clips (11.7%) have a sentence that also appears in those training rows. I scored the model separately on the two groups:

| Subset | Clips | WER | CER |
|---|---|---|---|
| Seen sentences | 1,739 | 0.147 | 0.0288 |
| Unseen sentences | 13,070 | 0.156 | 0.0282 |

The seen sentences score about 0.9 percentage points better on WER and slightly worse on CER, so I found no clear sign that overlap inflates the headline result. The seen group is much smaller and I did not test whether the gap is statistically meaningful, so I would not read much into a difference this size.

## Discussion

**Specialization beat generalist scale.** A smaller model fine-tuned specifically on Basque (wav2vec2-Basque) outperformed a much larger model trained across 99 languages (Whisper-large-v3) by a wide margin on both WER and CER. As a junior in this area, I read this as consistent with what the low-resource ASR literature describes, and as a concrete data point for choosing between fine-tuning a smaller model and relying on a large multilingual one.

**Speed did not help the model with no Basque exposure.** SenseVoice-Small is built for speed (non-autoregressive decoding, per An et al. 2024), and that held up here. But its transcripts were largely unusable on Basque. In the one raw output I inspected, the language tag was `<|en|>`, so it may be treating the audio as English. I only looked at that single clip, so this is a hypothesis, not a finding. This is in the same direction as the GigaSpeechBench (2026) observation that models can degrade a lot outside their trained distribution, here in an extreme case where the language was never in training.

**The shape of the errors differs, not just the totals.** For the three Basque-supporting models, substitutions make up roughly 70 to 80% of errors and deletions under 10%. SenseVoice-Small is different: deletions are about 48% of its errors. Substitutions are still its largest category, so I would describe it as an unusually high share of dropped words, not as deletion-dominated.

**Whisper-large-v3 insertions are unexplained.** Its insertions grew faster than the sample did (705 at 2,000 clips, 9,485 at 14,801) and its CER rose from 0.079 to 0.111. One possible explanation is hallucinated or repeated text on some clips, which I have not checked. Reading the clips with the most insertions would be the next step.

**Caveat on the wav2vec2-Basque result.** It was fine-tuned on `HiTZ/composite_corpus_eu_v2.1`. About 11.7% of test sentences also appear in the training rows I checked (see the overlap check above), but the model scored only slightly better on those than on unseen sentences, so overlap does not seem to explain its lead. The training data may still resemble Common Voice in recording style and sentence domains, so this is partly a comparison of a model tuned on Basque-like data against general models.

## Limitations

- Test set is Common Voice's scripted, standard-register speech. It does not cover dialectal or spontaneous Basque, where prior work shows performance drops.
- Train/test overlap for the own wav2vec2 model was checked at sentence level only, against the first 128,000 training rows. It does not detect identical audio, or near-duplicate sentences that differ after normalisation.- One run per model, no confidence intervals, and no per-accent breakdown yet (the `accents` column is available in the test set).
- Normalization is lowercase plus punctuation removal only, applied identically to all models. Different choices could shift WER somewhat.
- SenseVoice-Large is not publicly released, so only the Small variant was tested.

## License

Common Voice data is CC0. Re-hosting or re-sharing the raw dataset is prohibited per Mozilla's terms.
