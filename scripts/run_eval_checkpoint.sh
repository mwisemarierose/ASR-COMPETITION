#!/bin/bash
# Evaluate per-language dev WER for a checkpoint (no training).
#
# Usage:
#   CHECKPOINT=.../checkpoint-2000 ./scripts/run_eval_checkpoint.sh
#   CHECKPOINT=.../checkpoint-2000 FORCE_LANG_PROMPTS=1 ./scripts/run_eval_checkpoint.sh
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

export WORK_DIR="${WORK_DIR:-/project/community/rmwisene/pipeline_outputs}"
export HF_HOME="${HF_HOME:-/project/community/rmwisene/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a checkpoint or final/ directory}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$WORK_DIR/whisper_runs/eval_$(basename "$CHECKPOINT")}"
FORCE_LANG_PROMPTS="${FORCE_LANG_PROMPTS:-1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2400}"

cd ~/ASR-COMPETITION
mkdir -p "$HF_HOME" "$EVAL_OUTPUT"

PROMPT_ARGS=(--force-language-prompts)
if [[ "$FORCE_LANG_PROMPTS" == "0" ]]; then
  PROMPT_ARGS=(--no-force-language-prompts)
fi

echo "=== Eval checkpoint ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $EVAL_OUTPUT"

python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$EVAL_OUTPUT" \
  --model-name "$CHECKPOINT" \
  --eval-only \
  --max-eval-samples "$MAX_EVAL_SAMPLES" \
  --per-device-eval-batch-size 8 \
  --dataloader-num-workers 0 \
  --report-to none \
  "${PROMPT_ARGS[@]}"

echo "=== Done ==="
echo "WER JSON: $EVAL_OUTPUT/per_language_wer_eval_only.json"
