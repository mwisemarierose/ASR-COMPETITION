# Afrivoice Swahili ASR Pipeline

Audio-only pipeline for [DigitalUmuganda/Afrivoice_Swahili](https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_Swahili). Image prompt files in the source dataset are ignored.

---

## Table of contents

1. [Setup](#setup)
2. [Quick start (Orchard)](#quick-start-orchard)
3. [Pipeline steps](#pipeline-steps)
4. [All commands — `run_pipeline.py`](#all-commands--run_pipelinepy)
5. [All commands — `run_clean.py`](#all-commands--run_cleanpy)
6. [Domain & split filtering](#domain--split-filtering)
7. [CLI flag reference](#cli-flag-reference)
8. [Outputs & training data](#outputs--training-data)
9. [Help commands](#help-commands)

---

## Setup

```bash
git clone https://github.com/mwisemarierose/ASR-COMPETITION.git
cd ASR-COMPETITION

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No sudo needed — `imageio-ffmpeg` bundles a user-local ffmpeg for reading `.webm` files.

Verify ffmpeg:

```bash
python -c "from src.ffmpeg_setup import configure_ffmpeg; print(configure_ffmpeg())"
```

---

## Quick start (Orchard)

```bash
cd ~/ASR-COMPETITION
source .venv/bin/activate   # or: conda activate asr-competition

export DATASET_ROOT=/project/community/rmwisene/datasets/Afrivoice_Swahili
export WORK_DIR=/project/community/rmwisene/pipeline_outputs

# IMPORTANT: use --work-dir to avoid home disk quota errors
python run_pipeline.py \
  --dataset-root $DATASET_ROOT \
  --work-dir $WORK_DIR \
  --domain agriculture \
  --split dev
```

### Disk quota error?

If you see `OSError: [Errno 122] Disk quota exceeded`, your **home directory is full**. The dataset is too large for `~/ASR-COMPETITION/data/`.

**Fix:** send all outputs to project/community storage:

```bash
mkdir -p /project/community/rmwisene/pipeline_outputs

python run_pipeline.py \
  --dataset-root $DATASET_ROOT \
  --work-dir /project/community/rmwisene/pipeline_outputs \
  --domain agriculture
```

`--work-dir` creates:

```text
/project/community/rmwisene/pipeline_outputs/
├── extracted/     # unpacked .webm from tar.xz
├── cleaned/       # cleaned manifests
├── processed/     # 16 kHz WAV files
├── features/      # log-mel .npy files
└── statistics/    # JSON reports
```

Clean up failed runs in home:

```bash
rm -rf ~/ASR-COMPETITION/data/extracted
rm -rf ~/ASR-COMPETITION/data/cleaned
rm -rf ~/ASR-COMPETITION/data/processed
rm -rf ~/ASR-COMPETITION/outputs
```

### CPU time limit exceeded?

If you see `CPU time limit exceeded (core dumped)` on `orchard-login-001`, you hit the **login node time limit**. `agriculture/train` has **131,000+ files** — too heavy for the login node.

**Good news:** if extraction already finished, you do **not** need to re-extract.

**Recovery — finish clean only (fast):**

```bash
export DATASET_ROOT=/project/community/rmwisene/datasets/Afrivoice_Swahili
export WORK_DIR=/project/community/rmwisene/pipeline_outputs

python run_pipeline.py \
  --dataset-root $DATASET_ROOT \
  --work-dir $WORK_DIR \
  --domain agriculture --split train \
  --step clean \
  --skip-extract \
  --skip-audio-check \
  --skip-alignment-check
```

| Flag | Why |
|------|-----|
| `--skip-extract` | Extraction already done (131,247 files) |
| `--skip-audio-check` | Don't open every `.webm` — use manifest duration |
| `--skip-alignment-check` | Skip slow alignment checks on train |

**For heavy steps (preprocess, extract), use a compute node:**

```bash
cd ~/ASR-COMPETITION
mkdir -p logs
git pull   # get scripts/orchard_pipeline.slurm

sbatch scripts/orchard_pipeline.slurm agriculture train clean
sbatch scripts/orchard_pipeline.slurm agriculture train preprocess
sbatch scripts/orchard_pipeline.slurm agriculture train extract
```

Check job status:

```bash
squeue -u $USER
tail -f logs/asr-pipeline-*.out
```

---

## Pipeline steps

| Step | Command flag | What it does |
|------|-------------|--------------|
| 1. Clean | `--step clean` | Extract audio tar.xz, filter bad rows, normalize transcripts |
| 2. Preprocess | `--step preprocess` | Convert `.webm` → 16 kHz mono `.wav` |
| 3. Extract | `--step extract` | Generate 80-bin log-mel spectrograms (`.npy`) |
| 4. Validate | `--step validate` | Check all `.npy` feature files |

Default (no `--step`) runs all 4 steps in order.

---

## All commands — `run_pipeline.py`

### Show help

```bash
python run_pipeline.py --help
```

### Full pipeline — everything

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT
```

### Full pipeline — one domain (train + dev + test)

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture
python run_pipeline.py --dataset-root $DATASET_ROOT --domain education
python run_pipeline.py --dataset-root $DATASET_ROOT --domain financial
python run_pipeline.py --dataset-root $DATASET_ROOT --domain government
python run_pipeline.py --dataset-root $DATASET_ROOT --domain health
```

### Full pipeline — one split across all domains

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --split dev
python run_pipeline.py --dataset-root $DATASET_ROOT --split train
python run_pipeline.py --dataset-root $DATASET_ROOT --split test
```

### Full pipeline — one domain + one split

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --split dev
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --split train
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --split test
```

### Step by step — one domain

```bash
python run_pipeline.py --step clean      --dataset-root $DATASET_ROOT --domain agriculture
python run_pipeline.py --step preprocess --domain agriculture
python run_pipeline.py --step extract    --domain agriculture
python run_pipeline.py --step validate   --domain agriculture
```

### Step by step — one split

```bash
python run_pipeline.py --step clean      --dataset-root $DATASET_ROOT --domain agriculture --split dev
python run_pipeline.py --step preprocess --domain agriculture --split dev
python run_pipeline.py --step extract    --domain agriculture --split dev
python run_pipeline.py --step validate   --domain agriculture --split dev
```

### Multiple steps at once

```bash
python run_pipeline.py --step clean --step preprocess \
  --dataset-root $DATASET_ROOT --domain agriculture --split dev
```

### Verify only (no output files written)

```bash
python run_pipeline.py --verify-only \
  --dataset-root $DATASET_ROOT --domain agriculture --split dev
```

### Fast clean (skip opening every .webm)

```bash
python run_pipeline.py --step clean \
  --dataset-root $DATASET_ROOT --domain agriculture \
  --skip-audio-check --skip-alignment-check
```

### Reuse extracted audio cache (second run)

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --skip-extract
```

### Force re-extract tar.xz archives

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --force-extract
```

### Debug with limited rows

```bash
python run_pipeline.py --dataset-root $DATASET_ROOT --domain agriculture --split dev --max-records 100
```

### Local smoke test (no real dataset needed)

```bash
python run_pipeline.py \
  --dataset-root tests/fixtures \
  --extract-cache-root tests/fixtures_extracted \
  --domain agriculture --split dev \
  --skip-audio-check
```

---

## All commands — `run_clean.py`

Cleaning only (step 1). Use `run_pipeline.py` for the full pipeline.

### Show help

```bash
python run_clean.py --help
```

### Clean one split

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --split dev
```

### Clean whole domain (train + dev + test)

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture
```

### Clean all domains

```bash
python run_clean.py --dataset-root $DATASET_ROOT
```

### Verify only

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --verify-only
```

### Dry run (verify only, alias)

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --dry-run
```

### Skip audio file checks

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --skip-audio-check
```

### Custom duration filters

```bash
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --min-duration 2.0
python run_clean.py --dataset-root $DATASET_ROOT --domain agriculture --max-duration 120
```

### Custom output / cache paths

```bash
python run_clean.py \
  --dataset-root $DATASET_ROOT \
  --output-root /path/to/cleaned \
  --extract-cache-root /path/to/extracted \
  --domain agriculture
```

---

## Domain & split filtering

| Flags | What runs |
|-------|-----------|
| *(none)* | All 5 domains × 3 splits (15 folders) |
| `--domain agriculture` | agriculture train + dev + test |
| `--split dev` | dev for all 5 domains |
| `--domain agriculture --split dev` | agriculture dev only |

Available domains: `agriculture`, `education`, `financial`, `government`, `health`

Available splits: `train`, `dev`, `test`

---

## CLI flag reference

### `run_pipeline.py` flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--work-dir` | path | none | **Put all large outputs here** (use on Orchard!) |
| `--dataset-root` | path | `data/raw/Afrivoice_Swahili` | Path to Afrivoice_Swahili root |
| `--output-root` | path | `data/cleaned` | Where cleaned manifests are written |
| `--extract-cache-root` | path | `data/extracted` | Where tar.xz audio is unpacked |
| `--processed-root` | path | `data/processed` | Where 16 kHz WAV files are written |
| `--features-dir` | path | `outputs/features` | Where log-mel `.npy` files are written |
| `--domain` | choice | all | `agriculture`, `education`, `financial`, `government`, `health` |
| `--split` | choice | all | `train`, `dev`, `test` |
| `--step` | choice (repeatable) | all steps | `clean`, `preprocess`, `extract`, `validate` |
| `--verify-only` | flag | off | Verify manifests only, no output |
| `--skip-audio-check` | flag | off | Use manifest duration instead of opening `.webm` |
| `--skip-extract` | flag | off | Use existing extracted cache only |
| `--force-extract` | flag | off | Re-unpack tar.xz even if cache exists |
| `--skip-alignment-check` | flag | off | Skip transcript-audio alignment filter |
| `--max-records` | int | none | Limit rows per split (debugging) |

### `run_clean.py` flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset-root` | path | `data/raw/Afrivoice_Swahili` | Path to Afrivoice_Swahili root |
| `--output-root` | path | `data/cleaned` | Where cleaned manifests are written |
| `--domain` | choice | all | Filter by domain |
| `--split` | choice | all | Filter by split |
| `--verify-only` | flag | off | Verify only, no cleaned manifest |
| `--dry-run` | flag | off | Same as `--verify-only` |
| `--skip-audio-check` | flag | off | Use manifest duration instead of opening `.webm` |
| `--extract-cache-root` | path | `data/extracted` | Where tar.xz audio is unpacked |
| `--skip-extract` | flag | off | Use existing extracted cache only |
| `--force-extract` | flag | off | Re-unpack tar.xz even if cache exists |
| `--min-duration` | float | `2.0` | Drop clips shorter than this (seconds) |
| `--max-duration` | float | none | Optional upper duration limit (seconds) |
| `--max-records` | int | none | Limit rows per split (debugging) |

### Environment variable

```bash
export DATASET_ROOT=/project/community/rmwisene/datasets/Afrivoice_Swahili
# Used as default when --dataset-root is not passed
```

---

## Outputs & training data

### Folder layout after a full run

```text
data/extracted/<domain>/<split>/audio/*.webm     # unpacked from tar.xz
data/cleaned/<domain>/<split>/manifest_cleaned.jsonl
data/processed/<domain>/<split>/audio/*.wav       # 16 kHz mono — USE FOR TRAINING
data/processed/<domain>/<split>/manifest_processed.jsonl
outputs/features/<domain>/<split>/*.npy           # 80-bin log-mel — USE FOR TRAINING
outputs/statistics/*.json                         # reports only (not for training)
```

### Training with WAV + transcript (Whisper, Conformer, etc.)

```text
data/processed/agriculture/train/manifest_processed.jsonl
data/processed/agriculture/dev/manifest_processed.jsonl
data/processed/agriculture/test/manifest_processed.jsonl
data/processed/agriculture/train/audio/*.wav
```

Key columns: `audio_path`, `transcript`

### Training with precomputed features

```text
outputs/features/agriculture/train_features.tsv
outputs/features/agriculture/dev_features.tsv
outputs/features/agriculture/test_features.tsv
```

Key columns: `feature_path`, `transcript`, `feature_shape`

### Reports (inspection only)

```text
outputs/statistics/cleaning_report_<domain>.json
outputs/statistics/preprocessing_report.json
outputs/statistics/feature_extraction_report.json
outputs/statistics/feature_validation_report.json
```

---

## Dataset structure (input)

```text
Afrivoice_Swahili/
├── agriculture_swahili_dev/
│   ├── audio/
│   │   ├── audio_0.tar.xz
│   │   └── audio_1.tar.xz
│   ├── manifest_0.jsonl
│   └── manifest_1.jsonl
├── agriculture_swahili_train/
├── agriculture_swahili_test/
├── education_swahili_dev/
... (5 domains × 3 splits)
```

After extraction, audio files are `.webm` with hash-like names:

```text
MsYzAS3P092NBPdEBTMa.webm
xadnlONijhyVvN2USCkF.webm
```

---

## Cleaning rules

| Check | Action |
|-------|--------|
| Empty transcript | Removed |
| Missing audio file | Removed |
| Corrupt audio | Removed |
| Duration < 2 seconds | Removed |
| Duplicate `key` | Removed |
| Misaligned transcript/audio | Removed |
| Duration > 60 seconds | **Kept** (no upper limit) |

---

## Help commands

```bash
# Full pipeline help
python run_pipeline.py --help

# Cleaning-only help
python run_clean.py --help

# Check ffmpeg availability
python -c "from src.ffmpeg_setup import ffmpeg_status; print(ffmpeg_status())"

# List available domain/split folders
python -c "
from src.config import PipelineConfig
from src.discovery import DatasetDiscovery
d = DatasetDiscovery(PipelineConfig(dataset_root='$DATASET_ROOT'))
print(d.list_available_splits())
"
```

---

## Requirements implemented

| Task | Step |
|------|------|
| Remove empty transcripts | clean |
| Remove audio shorter than 2 seconds | clean |
| Verify transcript-audio alignment | clean |
| Resample all audio to 16 kHz | preprocess |
| Convert transcripts to lowercase | clean |
| Standardize transcript formatting | clean |
| Generate log-mel spectrogram features | extract |
| Configure 80 features per frame | extract |
| Validate feature generation | validate |
