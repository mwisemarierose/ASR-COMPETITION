#!/bin/bash
# Submit submission generation on Orchard GPU.
#
# Usage:
#   ./scripts/submit_generate_submission.sh
#
# Custom checkpoint + faster settings:
#   MODEL_DIR=/project/.../checkpoint-500 \
#   OUTPUT=/project/.../submission_step500.csv \
#   BATCH_SIZE=32 AUDIO_WORKERS=16 SLURM_CPUS=16 \
#   ./scripts/submit_generate_submission.sh
set -euo pipefail

export BATCH_SIZE="${BATCH_SIZE:-16}"
export AUDIO_WORKERS="${AUDIO_WORKERS:-8}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-128G}"

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/run_generate_submission.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --job-name=whisper-submission \
  --output=logs/whisper-submission-%j.out \
  --error=logs/whisper-submission-%j.err \
  --time=12:00:00 \
  --cpus-per-task="$SLURM_CPUS" \
  --mem="$SLURM_MEM" \
  --gres=gpu:1 \
  scripts/run_generate_submission.sh)

echo "Submitted submission generation"
echo "Job ID:  $JOBID"
echo "CPUs:    $SLURM_CPUS (AUDIO_WORKERS=$AUDIO_WORKERS, BATCH_SIZE=$BATCH_SIZE, MEM=$SLURM_MEM)"
echo "Model:   ${MODEL_DIR:-/project/community/rmwisene/pipeline_outputs/whisper_runs/multilingual_job_125891/checkpoint-12500}"
echo "Output:  ${OUTPUT:-/project/community/rmwisene/pipeline_outputs/whisper_runs/submission_checkpoint-12500.csv}"
echo ""
echo "Watch log:"
echo "  tail -f logs/whisper-submission-${JOBID}.out"
