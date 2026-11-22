#!/usr/bin/env bash
set -uo pipefail
cd "${MEM_PROJECT_ROOT:?}"
EXP="$MEM_PROJECT_ROOT/experiments/full-run-v2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEM0_TELEMETRY=false
status() { printf '%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" >> "$EXP/queue-status.tsv"; }
status queue started
# Existing retrieval job must finish successfully; no automatic tuning on held-out output.
while kill -0 "${RETRIEVAL_PID:?}" 2>/dev/null; do sleep 5; done
if [[ ! -f "$EXP/retrieval/complete.json" ]]; then status retrieval failed; exit 1; fi
status retrieval complete
status dense_pick running
scripts/bootstrap.sh python -u "$EXP/run_rerank_control.py" > "$EXP/dense-pick.log" 2>&1
if [[ "$?" -ne 0 ]]; then status dense_pick failed; exit 1; fi
status dense_pick complete
for baseline in mem0 amem; do
 status "$baseline" running
 scripts/bootstrap.sh python -u "$EXP/run_baseline_shards.py" --method "$baseline" --workers 4 > "$EXP/$baseline-full.log" 2>&1
 code=$?
 status "$baseline" "$code"
done
if [[ ! -f "$EXP/baselines/mem0-all125.json" || ! -f "$EXP/baselines/amem-all125.json" ]]; then
 status baselines incomplete
 exit 1
fi
status answers running
scripts/bootstrap.sh python -u "$EXP/evaluate_answers.py" > "$EXP/answers.log" 2>&1
code=$?
status answers "$code"
if [[ "$code" -eq 0 ]]; then status queue complete; else exit "$code"; fi
