#!/bin/bash
# Submit multilingual one-model training on Orchard.
#
# Usage:
#   ./scripts/submit_multilingual_train.sh
#   BALANCE=none ./scripts/submit_multilingual_train.sh      # all 970k clips
#   EPOCHS=1 ./scripts/submit_multilingual_train.sh           # one epoch (default)
#   EPOCHS=3 ./scripts/submit_multilingual_train.sh           # three epochs
set -euo pipefail

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/train_multilingual.sh

JOBID=$(sbatch --parsable \
  --job-name=whisper-multilingual \
  --output=logs/whisper-multilingual-%j.out \
  --error=logs/whisper-multilingual-%j.err \
  --time=10:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/train_multilingual.sh)

echo "Submitted multilingual training (one model, all 6 languages)"
echo "Job ID:  $JOBID"
echo "Balance: ${BALANCE:-cap} (MAX_PER_LANGUAGE=${MAX_PER_LANGUAGE:-80000})"
echo "Epochs:  ${EPOCHS:-1}"
echo ""
echo "Watch log:"
echo "  tail -f logs/whisper-multilingual-${JOBID}.out"
echo ""
echo "W&B: project asr-competition, group multilingual-one-model"
