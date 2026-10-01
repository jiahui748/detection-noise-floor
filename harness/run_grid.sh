#!/bin/bash
# Experiment 2b dose grid: full-model perturbation divergence with trajectory snapshots.
#
# Ordering is deliberate: seed 0 across all doses runs FIRST, so the dose-response
# curve exists after ~80 min instead of after the whole grid. Seeds 1 and 2 then
# supply the across-seed spread. Any run whose meta.json already exists is skipped,
# so the grid is resumable if the card goes away.
set -u
ROOT=<REMOTE-ROOT>
PY=$ROOT/envs/mw/bin/python
LOG=$ROOT/pr_exp/logs/grid.log
mkdir -p "$ROOT/pr_exp/logs" "$ROOT/pr_exp/result" "$ROOT/pr_exp/snap"
export PYTHONPATH=$ROOT/mw/RTDETR-main/RTDETR-main:$ROOT/mw:$ROOT

echo "=== GRID START $(date) ===" >> "$LOG"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "free disk before grid: ${free_gb} GB" >> "$LOG"

for seed in 0 1 2; do
  for eps in 0 1e-9 1e-8 1e-6 1e-4; do
    name="traj_e${eps}_s${seed}"
    meta="$ROOT/pr_exp/snap/$name/meta.json"
    if [ -f "$meta" ]; then
      echo "SKIP $name (already complete)" >> "$LOG"
      continue
    fi
    # guard: a run needs ~0.5 GB of snapshots; stop cleanly rather than fill the disk
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-0}" -lt 3 ]; then
      echo "ABORT $name: only ${free_gb} GB free" >> "$LOG"
      break 2
    fi
    echo "--- $name START $(date) ---" >> "$LOG"
    t0=$(date +%s)
    $PY -u "$ROOT/pr_exp/exp2b_traj.py" --eps "$eps" --seed "$seed" --epochs 3 \
        --name "$name" >> "$LOG" 2>&1
    rc=$?
    echo "--- $name EXIT rc=$rc wall=$(( ($(date +%s)-t0)/60 ))min $(date) ---" >> "$LOG"
  done
done

echo "=== GRID COMPLETE $(date) ===" >> "$LOG"
