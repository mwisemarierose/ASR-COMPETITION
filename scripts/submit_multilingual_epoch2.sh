#!/bin/bash
# Submit epoch-2 multilingual training (language prompts).
#
# Loads epoch-1 weights from INIT_FROM; does NOT resume optimizer state.
#
# Usage:
#   ./scripts/submit_multilingual_epoch2.sh
#
# Custom epoch-1 checkpoint:
#   INIT_FROM=/project/.../multilingual_job_125891/final ./scripts/submit_multilingual_epoch2.sh
#
# Equal balance (28k per language):
#   BALANCE=equal ./scripts/submit_multilingual_epoch2.sh
set -euo pipefail

export PHASE=epoch2
export INIT_FROM="${INIT_FROM:-/project/community/rmwisene/pipeline_outputs/whisper_runs/multilingual_job_125891/final}"
export BALANCE="${BALANCE:-cap}"
export EPOCHS="${EPOCHS:-1}"

exec "$(dirname "$0")/submit_multilingual_train.sh"
