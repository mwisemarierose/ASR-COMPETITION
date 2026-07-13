#!/bin/bash
# Submit submission generation on Orchard GPU.
#
# Usage:
#   ./scripts/submit_generate_submission.sh
#
# Custom checkpoint:
#   MODEL_DIR=/project/.../checkpoint-12500 OUTPUT=/project/.../submission_v1.csv ./scripts/submit_generate_submission.sh
set -euo pipefail

cd ~/ASR-COMPETITION
mkdir -p logs
chmod +x scripts/run_generate_submission.sh

JOBID=$(sbatch --parsable \
  --export=ALL \
  --job-name=whisper-submission \
  --output=logs/whisper-submission-%j.out \
  --error=logs/whisper-submission-%j.err \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  scripts/run_generate_submission.sh)

echo "Submitted submission generation"
echo "Job ID:  $JOBID"
echo "Model:   ${MODEL_DIR:-/project/community/rmwisene/pipeline_outputs/whisper_runs/multilingual_job_125891/checkpoint-12500}"
echo "Output:  ${OUTPUT:-/project/community/rmwisene/pipeline_outputs/whisper_runs/submission_checkpoint-12500.csv}"
echo ""
echo "Watch log:"
echo "  tail -f logs/whisper-submission-${JOBID}.out"
