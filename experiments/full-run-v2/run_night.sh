#!/usr/bin/env bash
set -euo pipefail
cd "${MEM_PROJECT_ROOT:?}"
EXP="$MEM_PROJECT_ROOT/experiments/full-run-v2"
exec 9>"$EXP/night-supervisor.lock"
flock -n 9 || exit 0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEM0_TELEMETRY=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY="$MEM_PROJECT_ROOT/.venv-baselines/bin/python"
export NIGHT_DEADLINE_UNIX="${NIGHT_DEADLINE_UNIX:-$(date -d '+8 hours' +%s)}"
printf '%s\n' "$NIGHT_DEADLINE_UNIX" > "$EXP/night-deadline-unix.txt"
"$MEM_PROJECT_ROOT/.venv/bin/python" "$EXP/night_data.py" > "$EXP/normalization.log" 2>&1
run_matrix() {
"$PY" "$EXP/night_worker.py" --seed
# One full-case calibration per dataset, then InMind full before larger datasets.
# A failed case is recorded; other cases and other methods continue.
pids=()
for method in dense hyde dense_pick properties mem0 amem rootmem rootmem_both; do
 "$PY" -u "$EXP/night_worker.py" --method "$method" > "$EXP/night-$method.log" 2>&1 &
 pids+=("$!")
done
for shard in 0 1; do
 "$PY" -u "$EXP/night_worker.py" --eval "$shard" > "$EXP/night-eval-$shard.log" 2>&1 &
 pids+=("$!")
done
"$PY" -u "$EXP/night_status.py" --watch > "$EXP/night-status.log" 2>&1 &
monitor=$!
printf '%s\n' "${pids[@]}" > "$EXP/night-worker-pids.txt"
printf '%s\n' "$monitor" >> "$EXP/night-worker-pids.txt"
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
kill "$monitor" || true
"$PY" "$EXP/night_status.py" > "$EXP/night-final-status.log"
}
run_matrix
"$PY" "$EXP/run_extension_b.py" > "$EXP/lucid-b-extension.log" 2>&1 || true
if [[ -f "$EXP/lucid-b-ready" ]]; then run_matrix; fi
exit "$code"
