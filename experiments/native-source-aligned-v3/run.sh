#!/usr/bin/env bash
set -euo pipefail
cd "${MEM_PROJECT_ROOT:?}"
EXP="$PWD/experiments/native-source-aligned-v3"
exec 9>"$EXP/run.lock"
flock -n 9 || exit 0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export NATIVE_DEADLINE="${NATIVE_DEADLINE:-$(date -d '+7 hours' +%s)}"
export NIGHT_DEADLINE_UNIX="$NATIVE_DEADLINE"
printf '%s\n' "$NATIVE_DEADLINE" > "$EXP/deadline.txt"
exec .venv-baselines/bin/python -u "$EXP/supervisor.py"
