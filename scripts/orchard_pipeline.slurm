#!/bin/bash
#SBATCH --job-name=asr-pipeline
#SBATCH --partition=general
#SBATCH --output=logs/asr-pipeline-%j.out
#SBATCH --error=logs/asr-pipeline-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Orchard / SLURM: run the pipeline on a compute node, not the login node.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/orchard_pipeline.slurm agriculture train clean
#   sbatch scripts/orchard_pipeline.slurm agriculture train preprocess
#   sbatch scripts/orchard_pipeline.slurm agriculture train
#
# Arguments:
#   $1 = domain   (agriculture, education, financial, government, health)
#   $2 = split      (train, dev, test) — optional
#   $3 = step       (clean, preprocess, extract) — optional

set -euo pipefail

DOMAIN="${1:-agriculture}"
SPLIT="${2:-}"
STEP="${3:-}"

export DATASET_ROOT=/project/community/rmwisene/datasets/Afrivoice_Swahili
export WORK_DIR=/project/community/rmwisene/pipeline_outputs

cd ~/ASR-COMPETITION
source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

mkdir -p logs
mkdir -p "$WORK_DIR"

CMD=(python run_pipeline.py
  --dataset-root "$DATASET_ROOT"
  --work-dir "$WORK_DIR"
  --domain "$DOMAIN"
)

if [[ -n "$SPLIT" ]]; then
  CMD+=(--split "$SPLIT")
fi

if [[ -n "$STEP" ]]; then
  CMD+=(--step "$STEP")
fi

# train splits are huge — skip per-file audio probes during clean (use manifest duration)
if [[ "$SPLIT" == "train" || -z "$SPLIT" ]]; then
  CMD+=(--skip-audio-check --skip-alignment-check --skip-verify)
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 1 ]]; then
  CMD+=(--workers "$SLURM_CPUS_PER_TASK")
fi

# Reuse extracted tar.xz only when THIS domain/split cache is complete.
EXTRACT_MARKER="$WORK_DIR/extracted/$DOMAIN/${SPLIT:-train}/audio/.extracted.json"
if [[ -n "$SPLIT" && -f "$EXTRACT_MARKER" ]]; then
  CMD+=(--skip-extract)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
