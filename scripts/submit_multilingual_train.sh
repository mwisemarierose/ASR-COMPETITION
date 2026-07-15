#!/bin/bash
# Submit multilingual one-model training on Orchard (epoch 1 from scratch).
#
# Usage:
#   ./scripts/submit_multilingual_train.sh
#   BALANCE=none ./scripts/submit_multilingual_train.sh      # all clips
#   EPOCHS=1 ./scripts/submit_multilingual_train.sh           # one epoch (default)
#   EPOCHS=3 ./scripts/submit_multilingual_train.sh           # three epochs
#
# Epoch 2 (separate script — loads a checkpoint, language prompts):
#   ./scripts/submit_multilingual_epoch2.sh
#
# Resume (must reuse the original job's output dir):
#   OUTPUT_DIR=/project/.../multilingual_job_126124 RESUME=1 DATALOADER_WORKERS=0 ./scripts/submit_multilingual_train.sh
set -euo pipefail

export PHASE=epoch1
export EPOCHS="${EPOCHS:-1}"
SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/train_multilingual.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --partition="$SLURM_PARTITION" \
  --job-name=whisper-multilingual \
  --output=logs/whisper-multilingual-%j.out \
  --error=logs/whisper-multilingual-%j.err \
  --time="$SLURM_TIME" \
  --cpus-per-task=64 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/train_multilingual.sh)

echo "Submitted multilingual training (one model, all 6 languages)"
echo "Phase:   ${PHASE} (whisper-small from scratch)"
echo "Job ID:  $JOBID"
echo "Partition: $SLURM_PARTITION (time=$SLURM_TIME)"
echo "Balance: ${BALANCE:-cap} (MAX_PER_LANGUAGE=${MAX_PER_LANGUAGE:-80000})"
echo "Epochs:  ${EPOCHS:-1}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  echo "Output:  $OUTPUT_DIR"
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  echo "Resume:  yes (latest checkpoint in OUTPUT_DIR)"
fi
echo ""
echo "Watch log:"
echo "  tail -f logs/whisper-multilingual-${JOBID}.out"
echo ""
echo "W&B: project asr-competition, group multilingual-one-model"
