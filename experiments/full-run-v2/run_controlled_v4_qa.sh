#!/usr/bin/env bash
set -euo pipefail
cd "${MEM_PROJECT_ROOT:?}"
EXP="$MEM_PROJECT_ROOT/experiments/full-run-v2"
exec 9>"$EXP/controlled-distractors-v4/qa.worker.lock"
flock -n 9 || exit 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NIGHT_DEADLINE_UNIX="$(date -d '+2 hours' +%s)"
"$MEM_PROJECT_ROOT/.venv-baselines/bin/python" -u "$EXP/controlled_distractors_qa_v4.py"
