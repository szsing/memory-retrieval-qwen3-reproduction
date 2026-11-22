#!/usr/bin/env bash
set -euo pipefail
cd "${MEM_PROJECT_ROOT:?}"
EXP="$PWD/experiments/original-qwen-night-v1"
exec 9>"$EXP/supervisor.lock"
flock -n 9 || exit 0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export NIGHT_DEADLINE_UNIX="${NIGHT_DEADLINE_UNIX:-$(date -d '+8 hours' +%s)}"
printf '%s\n' "$NIGHT_DEADLINE_UNIX" > "$EXP/deadline-unix.txt"
.venv/bin/python "$EXP/preflight.py" > "$EXP/preflight.log" 2>&1
.venv/bin/python -u "$EXP/supervisor.py"
