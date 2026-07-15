#!/bin/bash
# Remap legacy composite submission IDs to Parquet id values (CPU only, no GPU).
#
# Usage:
#   ./scripts/run_remap_submission_ids.sh
#
# Custom paths:
#   INPUT=/project/.../submission_checkpoint-2500_job126124.csv \
#   OUTPUT=/project/.../submission_fixed_ids.csv \
#   ./scripts/run_remap_submission_ids.sh
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

export KAGGLE_TEST_ROOT="${KAGGLE_TEST_ROOT:-/project/community/rmwisene/datasets/anv-test-data-nt}"
INPUT="${INPUT:-/project/community/rmwisene/pipeline_outputs/whisper_runs/submission_checkpoint-2500_job126124.csv}"
OUTPUT="${OUTPUT:-/project/community/rmwisene/pipeline_outputs/whisper_runs/submission_checkpoint-2500_job126124_fixed_ids.csv}"

cd ~/ASR-COMPETITION

echo "=== Remap submission IDs ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Test:   $KAGGLE_TEST_ROOT"

python scripts/remap_submission_ids.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --kaggle-test-root "$KAGGLE_TEST_ROOT"

echo "=== Done ==="
echo "Upload: $OUTPUT"
wc -l "$OUTPUT"
