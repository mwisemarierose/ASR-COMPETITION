#!/bin/bash
# Generate competition submission CSV from a Whisper checkpoint.
#
# Usage:
#   MODEL_DIR=.../checkpoint-12500 OUTPUT=.../submission_v1.csv ./scripts/run_generate_submission.sh
#   ./scripts/run_generate_submission.sh --dry-run
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

export WORK_DIR="${WORK_DIR:-/project/community/rmwisene/pipeline_outputs}"
export HF_HOME="${HF_HOME:-/project/community/rmwisene/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${MODEL_DIR:-$WORK_DIR/whisper_runs/multilingual_job_125891/checkpoint-12500}"
OUTPUT="${OUTPUT:-$WORK_DIR/whisper_runs/submission_checkpoint-12500.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FORCE_LANG_PROMPTS="${FORCE_LANG_PROMPTS:-0}"
SWAHILI_SPLIT="${SWAHILI_SPLIT:-test}"
ANV_SPLIT="${ANV_SPLIT:-dev_test}"

cd ~/ASR-COMPETITION
mkdir -p "$HF_HOME"

PROMPT_ARGS=()
if [[ "$FORCE_LANG_PROMPTS" == "1" ]]; then
  PROMPT_ARGS=(--force-language-prompts)
else
  PROMPT_ARGS=(--no-force-language-prompts)
fi

echo "=== Generate submission ==="
echo "Model:  $MODEL_DIR"
echo "Output: $OUTPUT"
echo "Batch:  $BATCH_SIZE"

echo "Swahili split: $SWAHILI_SPLIT | Anv split: $ANV_SPLIT"

python scripts/generate_submission.py \
  --work-dir "$WORK_DIR" \
  --model-dir "$MODEL_DIR" \
  --output "$OUTPUT" \
  --swahili-split "$SWAHILI_SPLIT" \
  --anv-split "$ANV_SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --expected-rows 41733 \
  "${PROMPT_ARGS[@]}" \
  "$@"

echo "=== Done ==="
echo "Upload: $OUTPUT"
