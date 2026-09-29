#!/bin/bash
# RESUME script for the experiment 2b dose grid.
#
# Why this file exists rather than an edit to run_grid.sh:
#   run_grid.sh checks for completion at $ROOT/pr_exp/snap/$name/meta.json, but the
#   runs write to $ROOT/mw/pr_exp/snap/$name/meta.json -- /lwz is a symlink to
#   /root/autodl-tmp, so the script's default --snapdir of /lwz/mw/pr_exp/snap
#   resolves under .../mw/. The original guard therefore never fires and a restart
#   would redo runs that are already complete.
#   The running script must not be edited in place: bash reads a script
#   incrementally, so modifying a file that is still executing can corrupt it.
#
# A run is considered complete when its meta.json exists; meta.json is written after
# all snapshot epochs, so a run interrupted part-way is simply redone.
set -u
ROOT=/root/autodl-tmp
PY=$ROOT/envs/mw/bin/python
SNAPROOT=$ROOT/mw/pr_exp/snap          # <- corrected path
LOG=$ROOT/mw/pr_exp/logs/grid_resume.log
mkdir -p "$ROOT/mw/pr_exp/logs" "$ROOT/mw/pr_exp/result" "$SNAPROOT"
export PYTHONPATH=$ROOT/mw/RTDETR-main/RTDETR-main:$ROOT/mw:$ROOT

echo "=== RESUME START $(date) ===" >> "$LOG"
for seed in 0 1 2; do
  for eps in 0 1e-9 1e-8 1e-6 1e-4; do
    name="traj_e${eps}_s${seed}"
    if [ -f "$SNAPROOT/$name/meta.json" ]; then
      echo "SKIP $name (complete)" >> "$LOG"
      continue
    fi
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-0}" -lt 4 ]; then
      echo "ABORT: only ${free_gb} GB free" >> "$LOG"; exit 1
    fi
    echo "--- $name START $(date) ---" >> "$LOG"
    t0=$(date +%s)
    $PY -u "$ROOT/pr_exp/exp2b_traj.py" --eps "$eps" --seed "$seed" --epochs 3 \
        --name "$name" >> "$LOG" 2>&1
    echo "--- $name EXIT rc=$? wall=$(( ($(date +%s)-t0)/60 ))min ---" >> "$LOG"
  done
done
echo "=== RESUME COMPLETE $(date) ===" >> "$LOG"
