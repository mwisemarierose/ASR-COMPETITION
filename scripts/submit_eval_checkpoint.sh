#!/bin/bash
# Submit standalone checkpoint eval on a GPU (safe while training runs elsewhere).
#
# Usage:
#   CHECKPOINT=/project/.../multilingual_epoch2_126017/checkpoint-2000 ./scripts/submit_eval_checkpoint.sh
set -euo pipefail

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/run_eval_checkpoint.sh scripts/submit_eval_checkpoint.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --job-name=whisper-eval \
  --output=logs/whisper-eval-%j.out \
  --error=logs/whisper-eval-%j.err \
  --time=02:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/run_eval_checkpoint.sh)

echo "Submitted checkpoint eval"
echo "Job ID:       $JOBID"
echo "Checkpoint:   ${CHECKPOINT:-<unset>}"
echo "Watch log:"
echo "  tail -f logs/whisper-eval-${JOBID}.out"
