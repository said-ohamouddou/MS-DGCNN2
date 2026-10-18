#!/usr/bin/env bash
# Run every experiment of the paper in order (clean training, corrupted test, 100 epochs).
# Each script reuses the clean checkpoints of the earlier ones (results/paper_shared_v2).
# Re-running this file resumes: finished folds and evaluations are reused, not retrained.
#
# Usage:
#   bash run_all.sh            # progress is shown live in the terminal
# Full logs (with progress bars) are also written to logs/<step>.log and logs/run_all.log.
#
# Do not edit models/, utils/ or core_experiments/ while it runs: checkpoint keys
# include the code, so a change would make later scripts retrain instead of reuse.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs

# conda's activation scripts use unset variables, so relax `set -u` while activating
set +u
source /home/said/miniconda3/etc/profile.d/conda.sh
conda activate torch311
set -u

LOG=logs/run_all.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Terminal view: one line per epoch/event; tqdm progress bars and empty lines hidden
show() { stdbuf -oL tr '\r' '\n' | sed -u -n '/it\/s/d; /^[[:space:]]*$/d; s/^INFO:[^:]*://; p'; }

STEPS=(
  "experiment1_ablation.py"
  "experiment2_density_dropout.py"
  "experiment3_noise_sweep.py"
  "run_ablations.py"
  "run_ablations.py --outliers"
  "run_kscale_ablation.py"
)

log "=== START (python: $(which python), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)) ==="
START=$(date +%s)
i=0
for step in "${STEPS[@]}"; do
  i=$((i + 1))
  name=$(echo "$step" | tr ' ' '_' | sed 's/\.py//; s/--//')
  log "--- [$i/${#STEPS[@]}] running: $step  (full log: logs/$name.log)"
  t0=$(date +%s)
  if ! python -u core_experiments/$step 2>&1 | tee "logs/$name.log" | show; then
    log "!!! FAILED: $step after $(( ($(date +%s)-t0)/60 )) min (see logs/$name.log)"
    exit 1
  fi
  log "--- [$i/${#STEPS[@]}] done: $step in $(( ($(date +%s)-t0)/60 )) min"
done
log "=== ALL DONE in $(( ($(date +%s)-START)/3600 )) h $(( (($(date +%s)-START)%3600)/60 )) min ==="
