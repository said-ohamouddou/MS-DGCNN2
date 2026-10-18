#!/usr/bin/env bash
# Live view of run_all.sh (read-only; safe to start/stop at any time, Ctrl+C to quit).
#   bash watch_progress.sh

cd "$(dirname "$(readlink -f "$0")")"
TOTAL_TRAININGS=185   # unique models: Exp1 20 + baselines 15 + matched 10 + Table 7 50 + k sweep 90

while true; do
  clear
  echo "=== MS-DGCNN++ experiments  |  $(date '+%a %d %b %H:%M:%S') ==="
  echo
  echo "--- Steps (logs/run_all.log)"
  tail -n 8 logs/run_all.log 2>/dev/null || echo "run_all.sh has not started yet"
  echo
  current=$(ls -t logs/*.log 2>/dev/null | grep -v run_all.log | head -1)
  if [ -n "$current" ]; then
    echo "--- Current: $(basename "$current")"
    tr '\r' '\n' < "$current" | grep -E "Fold [0-9]+ epoch|Reusing fold|OA=|Error|Traceback|Encoding:|Density|Noise|Neighborhood" \
      | grep -v "it/s" | tail -n 6
  fi
  echo
  done_models=$(ls results/paper_shared_v2/*.pth 2>/dev/null | wc -l)
  done_evals=$(ls results/paper_shared_v2/evaluations/*.json 2>/dev/null | wc -l)
  echo "--- Trained models: ${done_models}/${TOTAL_TRAININGS}   Saved evaluations: ${done_evals}"
  echo "--- GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)"
  if ! pgrep -f "core_experiments/" > /dev/null; then
    echo; echo "No experiment process running (finished, stopped or failed: see logs/run_all.log)."
  fi
  sleep 10
done
