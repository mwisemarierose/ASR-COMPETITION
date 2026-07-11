#!/bin/bash
# Train ONE multilingual model on all 6 competition languages at once.
# This avoids catastrophic forgetting from sequential language-by-language training.
#
# Submit:
#   ./scripts/submit_multilingual_train.sh
#
# Balance modes (BALANCE env var):
#   cap    — cap each language (default 80k; good compromise)
#   equal  — same count per language (uses only 28k each)
#   none   — use all 970k clips (Swahili dominates ~54%)
#
# Resume:
#   OUTPUT_DIR=.../multilingual_v1_... RESUME=1 sbatch ... scripts/train_multilingual.sh
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asr-competition

export WORK_DIR=/project/community/rmwisene/pipeline_outputs
export HF_HOME=/project/community/rmwisene/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT="${WANDB_PROJECT:-asr-competition}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-multilingual-one-model}"
# W&B is optional. Key is loaded from gitignored scripts/orchard_secrets.sh if present.
unset WANDB_ENTITY
SECRETS_FILE="$(dirname "$0")/orchard_secrets.sh"
if [[ -f "$SECRETS_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$SECRETS_FILE"
fi

BALANCE="${BALANCE:-cap}"
MAX_PER_LANGUAGE="${MAX_PER_LANGUAGE:-80000}"

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/whisper_runs/multilingual_v1_${RUN_TAG}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-multilingual-v1-${RUN_TAG}}"

mkdir -p "$HF_HOME" "$WORK_DIR/whisper_runs" "$OUTPUT_DIR"
cd ~/ASR-COMPETITION

BALANCE_ARGS=(--balance-languages "$BALANCE")
if [[ "$BALANCE" == "cap" ]]; then
  BALANCE_ARGS+=(--max-samples-per-language "$MAX_PER_LANGUAGE")
fi

echo "=== Multilingual training (one model) ==="
echo "Balance mode: $BALANCE"
echo "Output: $OUTPUT_DIR"

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Dataset summary ==="
python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dry-run \
  "${BALANCE_ARGS[@]}"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume-from-checkpoint)
  echo "=== Resuming from latest checkpoint in $OUTPUT_DIR ==="
fi

python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-name openai/whisper-small \
  --num-train-epochs 3 \
  --learning-rate 1e-5 \
  --warmup-steps 1000 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --gradient-checkpointing \
  --eval-steps 2000 \
  --save-steps 2000 \
  --logging-steps 50 \
  --save-total-limit 3 \
  --dataloader-num-workers 0 \
  --report-to wandb tensorboard \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_RUN_GROUP" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --eval-all-languages \
  "${BALANCE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
echo "Competition model: $OUTPUT_DIR/final"
