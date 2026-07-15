#!/bin/bash
# Train ONE multilingual model on all 6 competition languages at once.
# This avoids catastrophic forgetting from sequential language-by-language training.
#
# Submit:
#   ./scripts/submit_multilingual_train.sh
#
# Balance modes (BALANCE env var):
#   cap    — cap each language (default 80k; good compromise)
#   equal  — same count per language (uses smallest language count)
#   none   — use all clips (Swahili dominates ~53%)
#
# Resume epoch 1 (same OUTPUT_DIR, same data):
#   OUTPUT_DIR=.../multilingual_job_125891 RESUME=1 ./scripts/submit_multilingual_train.sh
#
# Epoch 2 (load epoch-1 weights only — do NOT use RESUME=1):
#   PHASE=epoch2 INIT_FROM=/project/.../multilingual_job_125891/checkpoint-12500 ./scripts/submit_multilingual_epoch2.sh
#
# Epochs (EPOCHS env var):
#   EPOCHS=1 ./scripts/submit_multilingual_train.sh   # one epoch (~13k steps)
#   EPOCHS=3 ./scripts/submit_multilingual_train.sh   # full run (~40k steps)
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

PHASE="${PHASE:-epoch1}"
BALANCE="${BALANCE:-cap}"
MAX_PER_LANGUAGE="${MAX_PER_LANGUAGE:-80000}"
EPOCHS="${EPOCHS:-1}"

case "$PHASE" in
  epoch1)
    MODEL_NAME="${MODEL_NAME:-openai/whisper-small}"
    LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
    GRAD_ACCUM="${GRAD_ACCUM:-4}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    PER_LANG_EVAL_STEPS="${PER_LANG_EVAL_STEPS:-500}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
    LANG_PROMPT_ARGS=()
    ALIGN_ARGS=()
    ;;
  epoch2)
    MODEL_NAME="${MODEL_NAME:-${INIT_FROM:-$WORK_DIR/whisper_runs/multilingual_job_125891/checkpoint-12500}}"
    LEARNING_RATE="${LEARNING_RATE:-5e-6}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
    GRAD_ACCUM="${GRAD_ACCUM:-4}"
    SAVE_STEPS="${SAVE_STEPS:-500}"
    PER_LANG_EVAL_STEPS="${PER_LANG_EVAL_STEPS:-500}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
    LANG_PROMPT_ARGS=(--force-language-prompts)
    ALIGN_ARGS=()
    ;;
  *)
    echo "ERROR: unknown PHASE=$PHASE (expected epoch1 or epoch2)" >&2
    exit 1
    ;;
esac

# Stable output dir across SLURM requeues (set OUTPUT_DIR explicitly when resuming).
if [[ -z "${OUTPUT_DIR:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ "$PHASE" == "epoch2" ]]; then
    export OUTPUT_DIR="$WORK_DIR/whisper_runs/multilingual_epoch2_${SLURM_JOB_ID}"
  else
    export OUTPUT_DIR="$WORK_DIR/whisper_runs/multilingual_job_${SLURM_JOB_ID}"
  fi
fi

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
if [[ "$PHASE" == "epoch2" && -z "${OUTPUT_DIR:-}" ]]; then
  export OUTPUT_DIR="$WORK_DIR/whisper_runs/multilingual_epoch2_${RUN_TAG}"
else
  export OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/whisper_runs/multilingual_v1_${RUN_TAG}}"
fi
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-multilingual-${PHASE}-${SLURM_JOB_ID:-${RUN_TAG}}}"

mkdir -p "$HF_HOME" "$WORK_DIR/whisper_runs" "$OUTPUT_DIR"
cd ~/ASR-COMPETITION

BALANCE_ARGS=(--balance-languages "$BALANCE")
if [[ "$BALANCE" == "cap" ]]; then
  BALANCE_ARGS+=(--max-samples-per-language "$MAX_PER_LANGUAGE")
fi

echo "=== Multilingual training (one model) ==="
echo "Phase: $PHASE"
echo "Model: $MODEL_NAME"
echo "Balance mode: $BALANCE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Train batch / grad accum: $TRAIN_BATCH_SIZE / $GRAD_ACCUM (effective $((TRAIN_BATCH_SIZE * GRAD_ACCUM)))"
echo "DataLoader workers: $DATALOADER_WORKERS"
echo "Output: $OUTPUT_DIR"
echo "Checkpoints: every ${SAVE_STEPS:-500} steps (keep all; set SAVE_TOTAL_LIMIT=N to prune old ones)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "SLURM job: $SLURM_JOB_ID"
fi

SAVE_TOTAL_LIMIT_ARGS=()
if [[ -n "${SAVE_TOTAL_LIMIT:-}" ]]; then
  SAVE_TOTAL_LIMIT_ARGS=(--save-total-limit "$SAVE_TOTAL_LIMIT")
fi

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Dataset summary ==="
python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dry-run \
  "${BALANCE_ARGS[@]}"

RESUME_ARGS=()
if [[ "$PHASE" == "epoch1" ]]; then
  if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint)
    # Forked workers + parquet/ffmpeg decode can segfault after long runs; load in main process.
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
    echo "=== Resuming from latest checkpoint in $OUTPUT_DIR (dataloader workers=$DATALOADER_WORKERS) ==="
  elif compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
    RESUME_ARGS=(--resume-from-checkpoint)
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
    echo "=== Auto-resuming from latest checkpoint in $OUTPUT_DIR (SLURM requeue, dataloader workers=$DATALOADER_WORKERS) ==="
  fi
else
  echo "=== Epoch 2: loading weights from $MODEL_NAME (fresh optimizer, no RESUME) ==="
fi

python scripts/finetune_whisper.py \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --num-train-epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --gradient-checkpointing \
  --save-steps "$SAVE_STEPS" \
  --per-language-eval-steps "$PER_LANG_EVAL_STEPS" \
  --max-eval-samples 2400 \
  --logging-steps 50 \
  "${SAVE_TOTAL_LIMIT_ARGS[@]}" \
  --dataloader-num-workers "$DATALOADER_WORKERS" \
  --report-to wandb tensorboard \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_RUN_GROUP" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --eval-all-languages \
  "${BALANCE_ARGS[@]}" \
  "${LANG_PROMPT_ARGS[@]}" \
  "${ALIGN_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

echo "=== Done ==="
echo "Competition model: $OUTPUT_DIR/final"
