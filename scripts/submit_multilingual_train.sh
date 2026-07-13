#!/bin/bash
# Submit multilingual one-model training on Orchard.
#
# Usage:
#   ./scripts/submit_multilingual_train.sh
#   BALANCE=none ./scripts/submit_multilingual_train.sh      # all 970k clips
#   EPOCHS=1 ./scripts/submit_multilingual_train.sh           # one epoch (default)
#   EPOCHS=3 ./scripts/submit_multilingual_train.sh           # three epochs
#
# Resume (must reuse the original job's output dir):
#   OUTPUT_DIR=/project/.../multilingual_job_125891 RESUME=1 ./scripts/submit_multilingual_train.sh
set -euo pipefail

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/train_multilingual.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --job-name=whisper-multilingual \
  --output=logs/whisper-multilingual-%j.out \
  --error=logs/whisper-multilingual-%j.err \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/train_multilingual.sh)

echo "Submitted multilingual training (one model, all 6 languages)"
echo "Job ID:  $JOBID"
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
