#!/bin/bash
#SBATCH --job-name=anv-pipeline
#SBATCH --output=logs/anv-pipeline-%j.out
#SBATCH --error=logs/anv-pipeline-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Orchard / SLURM: run Anv-ke Parquet pipeline on a compute node.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/orchard_anv_pipeline.slurm kalenjin train unscripted clean
#   sbatch scripts/orchard_anv_pipeline.slurm kalenjin train unscripted extract
#   sbatch scripts/orchard_anv_pipeline.slurm dholuo dev scripted
#
# Arguments:
#   $1 = language  (kalenjin, dholuo, kikuyu, somali, maasai)
#   $2 = split       (train, dev, test, dev_test) — optional
#   $3 = style       (scripted, unscripted) — optional
#   $4 = step        (clean, extract) — optional

set -euo pipefail

LANGUAGE="${1:-kalenjin}"
SPLIT="${2:-}"
STYLE="${3:-}"
STEP="${4:-}"

export ANV_DATASET_ROOT=/project/community/rmwisene/datasets
export WORK_DIR=/project/community/rmwisene/pipeline_outputs

# Map language slug -> on-disk folder name
case "$LANGUAGE" in
  kalenjin) LANG_DIR=Kalenjin ;;
  dholuo|luo) LANG_DIR=Dholuo ;;
  kikuyu) LANG_DIR=Kikuyu ;;
  somali) LANG_DIR=Somali ;;
  maasai) LANG_DIR=Maasai ;;
  *)
    echo "Unknown language: $LANGUAGE"
    exit 1
    ;;
esac

cd ~/ASR-COMPETITION
source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

mkdir -p logs
mkdir -p "$WORK_DIR"

CMD=(python run_pipeline.py
  --dataset-type anv
  --language "$LANGUAGE"
  --dataset-root "$ANV_DATASET_ROOT/$LANG_DIR"
  --work-dir "$WORK_DIR"
)

if [[ -n "$SPLIT" ]]; then
  CMD+=(--split "$SPLIT")
fi

if [[ -n "$STYLE" ]]; then
  CMD+=(--style "$STYLE")
fi

if [[ -n "$STEP" ]]; then
  CMD+=(--step "$STEP")
fi

# Large train splits: trust CSV duration, skip per-clip decode during clean
if [[ "$SPLIT" == "train" || -z "$SPLIT" ]]; then
  CMD+=(--skip-audio-check --skip-verify)
fi

  # Extract always runs sequentially in code; workers only apply to clean.
  if [[ "$STEP" == "clean" && -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 1 ]]; then
    CMD+=(--workers "$SLURM_CPUS_PER_TASK")
  fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
