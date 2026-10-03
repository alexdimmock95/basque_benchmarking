# Basque ASR Benchmarking

Benchmarking multiple pretrained speech-to-text models on Basque (Euskara), comparing accuracy, character-level accuracy, and inference speed across models that officially support Basque versus models tested zero-shot.

## Motivation

Several papers report ASR results for Basque individually (e.g. fine-tuned Whisper variants, bilingual Spanish-Basque systems), but no existing work compares multiple modern ASR models on the same Basque test set under the same conditions. This project fills that gap, and additionally reports how non-Basque-supporting models perform when tested zero-shot on the language.

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
