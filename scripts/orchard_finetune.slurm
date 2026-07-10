#!/bin/bash
#SBATCH --job-name=whisper-ft
#SBATCH --output=logs/whisper-ft-%j.out
#SBATCH --error=logs/whisper-ft-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Orchard / SLURM: fine-tune Whisper on A100.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/orchard_finetune.slurm smoke
#   sbatch scripts/orchard_finetune.slurm swahili
#   sbatch scripts/orchard_finetune.slurm full
#
# Modes:
#   smoke   — agriculture only, 1k/source, 200 steps (validate pipeline today)
#   swahili — all 5 Swahili domains, dev eval
#   full    — all 6 languages, multilingual train

set -euo pipefail

MODE="${1:-smoke}"
export WORK_DIR=/project/community/rmwisene/pipeline_outputs
export HF_HOME="${HF_HOME:-/project/community/rmwisene/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

cd ~/ASR-COMPETITION
source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

mkdir -p logs
mkdir -p "$WORK_DIR/whisper_runs"
mkdir -p "$HF_HOME"

RUN_DIR="$WORK_DIR/whisper_runs/${MODE}_$(date +%Y%m%d_%H%M%S)"
CMD=(python scripts/finetune_whisper.py --work-dir "$WORK_DIR" --output-dir "$RUN_DIR")

case "$MODE" in
  smoke)
    CMD+=(
      --swahili-domains agriculture
      --no-anv
      --max-samples-per-source 1000
      --max-steps 200
      --eval-steps 50
      --save-steps 100
      --per-device-train-batch-size 8
      --gradient-accumulation-steps 2
    )
    ;;
  swahili)
    CMD+=(
      --no-anv
      --max-samples-per-source 5000
      --num-train-epochs 1
      --eval-steps 1000
      --save-steps 1000
      --per-device-train-batch-size 8
      --gradient-accumulation-steps 4
    )
    ;;
  full)
    CMD+=(
      --num-train-epochs 1
      --eval-steps 2000
      --save-steps 2000
      --per-device-train-batch-size 8
      --gradient-accumulation-steps 4
      --gradient-checkpointing
    )
    ;;
  *)
    echo "Unknown mode: $MODE (use smoke, swahili, or full)"
    exit 1
    ;;
esac

echo "Mode: $MODE"
echo "Output: $RUN_DIR"
echo "Running: ${CMD[*]}"
"${CMD[@]}"
