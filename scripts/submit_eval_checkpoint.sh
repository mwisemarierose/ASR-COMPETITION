#!/bin/bash
# Submit standalone checkpoint eval on a GPU (safe while training runs elsewhere).
#
# Usage:
#   CHECKPOINT=/project/.../multilingual_epoch2_126017/checkpoint-2000 ./scripts/submit_eval_checkpoint.sh
set -euo pipefail

SLURM_PARTITION="${SLURM_PARTITION:-general}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/run_eval_checkpoint.sh scripts/submit_eval_checkpoint.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --partition="$SLURM_PARTITION" \
  --job-name=whisper-eval \
  --output=logs/whisper-eval-%j.out \
  --error=logs/whisper-eval-%j.err \
  --time="$SLURM_TIME" \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/run_eval_checkpoint.sh)

echo "Submitted checkpoint eval"
echo "Job ID:       $JOBID"
echo "Partition:    $SLURM_PARTITION (time=$SLURM_TIME)"
echo "Checkpoint:   ${CHECKPOINT:-<unset>}"
echo "Watch log:"
echo "  tail -f logs/whisper-eval-${JOBID}.out"
