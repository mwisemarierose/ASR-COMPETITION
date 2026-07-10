#!/bin/bash
# Orchard smoke test: Whisper fine-tune on agriculture (1000 clips, 200 steps).
# Submit with:
#   sbatch --job-name=whisper-smoke --output=logs/whisper-smoke-%j.out \
#     --error=logs/whisper-smoke-%j.err --time=4:00:00 --cpus-per-task=8 \
#     --mem=64G --gres=gpu:1 scripts/smoke_agriculture.sh
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

export WORK_DIR=/project/community/rmwisene/pipeline_outputs
export HF_HOME=/project/community/rmwisene/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$WORK_DIR/whisper_runs"
cd ~/ASR-COMPETITION

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Dry run ==="
python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$WORK_DIR/whisper_runs/dry_run" \
  --swahili-domains agriculture \
  --no-anv \
  --dry-run

echo "=== Smoke test ==="
python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$WORK_DIR/whisper_runs/smoke_agriculture" \
  --swahili-domains agriculture \
  --no-anv \
  --max-samples-per-source 1000 \
  --max-steps 200 \
  --eval-steps 50 \
  --save-steps 100 \
  --logging-steps 10 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2

echo "=== Done ==="
