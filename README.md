# Afrivoice Swahili ASR Pipeline

Full terminal pipeline for [DigitalUmuganda/Afrivoice_Swahili](https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_Swahili).

## Requirements implemented

| Task | Step | Module |
|------|------|--------|
| Remove empty transcripts | clean | `RecordFilter` |
| Remove audio shorter than 2 seconds | clean | `AudioDurationValidator` |
| Verify transcript-audio alignment | clean | `AlignmentValidator` |
| Resample all audio to 16 kHz | preprocess | `AfrivoicePreprocessingPipeline` |
| Convert transcripts to lowercase | clean | `SwahiliTranscriptCleaner` |
| Standardize transcript formatting | clean | `SwahiliTranscriptCleaner` |
| Generate log-mel spectrogram features | extract | `LogMelFeatureExtractor` |
| Configure 80 features per frame | extract | `N_MELS = 80` |
| Validate feature generation | validate | `FeatureValidator` |
| Apply time masking | augment | `SpecAugmenter` |
| Apply frequency masking | augment | `SpecAugmenter` |

## Setup

```bash
cd "/Users/pinkm/Desktop/ASR COMPETITION"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No sudo needed — `requirements.txt` includes `imageio-ffmpeg`, which bundles a user-local ffmpeg binary for reading `.webm` files.

Optional alternatives if you prefer:

```bash
# conda (also no sudo, installs in your home directory)
conda install -c conda-forge ffmpeg
```

## Run full pipeline

```bash
python run_pipeline.py --dataset-root /path/to/Afrivoice_Swahili \
  --domain agriculture --split dev
```

## Run step by step

```bash
python run_pipeline.py --step clean --dataset-root /path/to/Afrivoice_Swahili --domain agriculture --split dev
python run_pipeline.py --step preprocess --domain agriculture --split dev
python run_pipeline.py --step extract --domain agriculture --split dev
python run_pipeline.py --step augment --domain agriculture
python run_pipeline.py --step validate --domain agriculture --split dev
```

## Outputs

After extraction, audio files are `.webm` (e.g. `MsYzAS3P092NBPdEBTMa.webm`). Preprocessing converts them to 16 kHz `.wav`.

```text
data/extracted/<domain>/<split>/audio/*.webm   # unpacked from tar.xz
data/cleaned/<domain>/<split>/manifest_cleaned.jsonl
data/processed/<domain>/<split>/audio/*.wav    # 16 kHz mono
data/processed/<domain>/<split>/manifest_processed.jsonl
outputs/features/<domain>/<split>/*.npy        # 80-bin log-mel
outputs/features/<domain>/train_augmented/       # SpecAugment copies
outputs/statistics/*.json                        # reports per step
```

## Local smoke test

```bash
python run_pipeline.py \
  --dataset-root tests/fixtures \
  --extract-cache-root tests/fixtures_extracted \
  --domain agriculture --split dev \
  --skip-audio-check
```

## Flags

| Flag | Purpose |
|------|---------|
| `--step` | Run one or more steps only |
| `--skip-audio-check` | Faster cleaning using manifest duration |
| `--skip-alignment-check` | Skip transcript-audio alignment filter |
| `--skip-extract` | Reuse extracted tar.xz cache |
| `--max-records N` | Limit rows per split for testing |
