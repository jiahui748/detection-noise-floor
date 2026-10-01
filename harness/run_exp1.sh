#!/usr/bin/env bash
# Experiment 1 -- grid over the deformable-attention sampler x strictness arms, on a SECOND
# architecture (HF DeformableDetrForObjectDetection, transformers 5.17.0) against
# VisDrone2019-DET at 640 px.
#
#   3 arms, same-seed (seed 0) repeats, because the quantity of interest is the run-to-run
#   jitter at a FIXED seed:
#
#     arm a  F.grid_sample, strict off                  6 repeats  <- the sigma estimate
#     arm c  deterministic gather sampler, strict off   3 repeats
#     arm d  deterministic gather sampler, strict on     2 repeats  <- expected bit-identical
#
#   Arm (d) is 2 repeats because bit-identity is established by two identical hashes; the
#   third was redundancy costing ~2.9 h against a hard 06:00 shutdown.
#   A run whose result JSON already exists is SKIPPED, so the grid is resumable.
#
# BATCH SIZE IS PER ARM, from measurement, because the gather sampler needs far more
# activation memory than F.grid_sample and cannot run at the same batch size:
#
#   arm a  bs=4  (peak 18.0 GiB, 5.06 min/epoch measured on the real grid)
#   arm c  bs=1  (peak 18.3 GiB, 19.83 min/epoch measured)   accum=4
#   arm d  bs=1  (peak 10.3 GiB, 57.24 min/epoch measured)   accum=4
#
#   The effective batch is 4 in every arm (bs x accum), so no arm sees a different number
#   of images per update.  Batch size is NOT the manipulated variable; sampler and
#   strictness are.  See EXP1_NOTES.md.
#
# INCREMENTAL HARVESTING: every result JSON is copied off the box the moment it lands, so
# no result ever exists only on the remote disk.  The weights stay remote -- the JSON
# carries the sha256, which is the evidence.
#
# Usage on the box (detached; STOP_AT refuses to start a run that cannot finish before the
# server's auto-shutdown):
#   RUN_REPS_D=2 RUN_STOP_AT="2026-09-29 05:15" \
#   setsid bash <REMOTE-ROOT>/pr_exp/run_exp1.sh \
#       > <REMOTE-ROOT>/pr_exp/exp1/logs/grid.out 2>&1 < /dev/null &
set -u
ROOT=<REMOTE-ROOT>
PY=$ROOT/envs/mw/bin/python
PR=$ROOT/pr_exp
BASE=$PR/exp1
RES=$BASE/result
LOGS=$BASE/logs
WTS=$BASE/weights
DATA=$ROOT/mw/VisDrone
mkdir -p "$RES" "$LOGS" "$WTS"

EPOCHS=${RUN_EPOCHS:-6}
SIZE=${RUN_SIZE:-640}
WORKERS=${RUN_WORKERS:-8}
SEED=${RUN_SEED:-0}
LIMIT=${RUN_LIMIT:-0}
PRETRAINED=${RUN_PRETRAINED:-SenseTime/deformable-detr}
MIN_FREE_GB=${RUN_MIN_FREE_GB:-3}
MIN_FREE_MIB=${RUN_MIN_FREE_MIB:-19000}   # largest per-arm peak, plus headroom
BATCH_A=${RUN_BATCH_A:-4}
BATCH_C=${RUN_BATCH_C:-1}
BATCH_D=${RUN_BATCH_D:-1}
EPOCHS_A=${RUN_EPOCHS_A:-$EPOCHS}
EPOCHS_C=${RUN_EPOCHS_C:-$EPOCHS}
EPOCHS_D=${RUN_EPOCHS_D:-3}
REPS_A=${RUN_REPS_A:-6}
REPS_C=${RUN_REPS_C:-3}
REPS_D=${RUN_REPS_D:-2}
STOP_AT=${RUN_STOP_AT:-""}
HARVEST_STAGE=${RUN_HARVEST_STAGE:-$BASE/harvest}
RR=$ROOT/pr_exp/rr.py
WINDEST='<WORKSPACE>\pr_exp\results\exp1'
HEAD_LR_MULT=${RUN_HEAD_LR_MULT:-10}
MAX_GRAD_NORM=${RUN_MAX_GRAD_NORM:-0}
ACCUM=${RUN_ACCUM:-4}

export PYTHONPATH="$ROOT/mw:$PR:$ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export USE_HUB_KERNELS=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

GRIDLOG=$LOGS/grid.log
say() { echo "$*" | tee -a "$GRIDLOG"; }

say "=== EXP1 GRID START $(date) pid=$$ seed=$SEED size=$SIZE ==="
say "    epochs a=$EPOCHS_A c=$EPOCHS_C d=$EPOCHS_D   repeats a=$REPS_A c=$REPS_C d=$REPS_D"
say "    batch a=$BATCH_A c=$BATCH_C d=$BATCH_D   effective batch $ACCUM updates per arm"
say "    max_grad_norm=$MAX_GRAD_NORM head_lr_mult=$HEAD_LR_MULT stop_at=${STOP_AT:-none}"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
say "    free disk: ${free_gb} GB   min required: ${MIN_FREE_GB} GB"

wait_for_gpu () {
  local need=$1 waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    [ -z "$free" ] && { sleep 30; continue; }
    if [ "$free" -ge "$need" ]; then
      [ "$waited" -gt 0 ] && say "GPU free ${free} MiB after waiting ${waited}s"
      return 0
    fi
    if [ $((waited % 300)) -eq 0 ]; then
      say "WAIT  gpu has ${free} MiB free, need ${need} MiB (waited ${waited}s)"
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

# copy every result JSON off the box.  Runs at the end of every run, so an interruption
# costs at most the run in flight, never the whole grid.
#
# Two layers, deliberately:
#   1. the STAGING copy below always works -- a redundant directory on the remote disk
#      (the workspace disk, which survives on AutoDL), refreshed after every single run.
#   2. a pull straight to the Windows workspace, which can only work if this box has a
#      python WITH paramiko on its PATH.  It does not: there is no `python` on PATH at
#      all, and the env python has no paramiko.  So this layer is expected to fail here
#      and the message says so rather than pretending otherwise; the local copy is made
#      from the Windows side out of the staging directory.
harvest () {
  mkdir -p "$HARVEST_STAGE"
  local n=0
  for f in "$RES"/*.json; do
    [ -f "$f" ] || continue
    cp -f "$f" "$HARVEST_STAGE/" && n=$((n + 1))
  done
  [ -f "$BASE/exp1_sigma.json" ] && cp -f "$BASE/exp1_sigma.json" "$HARVEST_STAGE/"
  say "HARVEST staged $n JSON -> $HARVEST_STAGE (workspace disk; pull from Windows)"
  if command -v python >/dev/null 2>&1; then
    local files=()
    for f in "$HARVEST_STAGE"/*.json; do [ -f "$f" ] && files+=("$f"); done
    if [ ${#files[@]} -gt 0 ]; then
      if python "$RR" get "${files[@]}" --dest "$WINDEST" >/dev/null 2>&1; then
        say "HARVEST pulled ${#files[@]} JSON to local workspace"
      else
        say "HARVEST local pull unavailable; staged copy is intact, pull it from Windows"
      fi
    fi
  else
    say "HARVEST no python on PATH; staged copy is intact, pull it from Windows"
  fi
}

run_one () {
  local arm=$1 rep=$2 bs=$3 eps=$4
  local name="exp1_${arm}_s${SEED}_r${rep}"
  local out="$RES/$name.json"
  local log="$LOGS/$name.log"

  if [ -f "$out" ]; then
    say "SKIP  $name  (result exists)"
    return 0
  fi
  if [ -n "$STOP_AT" ]; then
    if [ "$(date '+%F %T')" \> "$STOP_AT" ]; then
      say "ABORT $name: past stop_at=$STOP_AT, not starting a run that cannot finish"
      return 9
    fi
  fi
  local fg
  fg=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  if [ "${fg:-0}" -lt "$MIN_FREE_GB" ]; then
    say "ABORT $name: only ${fg} GB free (< ${MIN_FREE_GB})"
    return 9
  fi
  wait_for_gpu "$MIN_FREE_MIB" || return 9

  local extra=""
  [ "$LIMIT" != "0" ] && extra="--limit-train $LIMIT"
  say "--- $name START $(date) bs=$bs epochs=$eps ---"
  local t0; t0=$(date +%s)
  $PY -u "$PR/exp1_deformable.py" \
      --arm "$arm" --seed "$SEED" --repeat "$rep" \
      --epochs "$eps" --batch "$bs" --accum "$ACCUM" \
      --max-grad-norm "$MAX_GRAD_NORM" --head-lr-mult "$HEAD_LR_MULT" \
      --size "$SIZE" --workers "$WORKERS" \
      --data "$DATA" --out "$RES" --weights-dir "$WTS" \
      --pretrained "$PRETRAINED" $extra > "$log" 2>&1
  local rc=$?
  local wall=$(( ($(date +%s) - t0) / 60 ))
  say "--- $name EXIT rc=$rc wall=${wall}min $(date) ---"
  if [ $rc -ne 0 ]; then
    say "     tail of $log:"
    tail -n 25 "$log" | sed 's/^/     | /' | tee -a "$GRIDLOG"
  fi
  $PY "$PR/exp1_analyse.py" --result "$RES" >/dev/null 2>&1 || true
  harvest
  return $rc
}

# Each arm loops to its repeat count; a disk/stop refusal ends the whole grid cleanly.
stopped=0
for spec in "a $REPS_A $BATCH_A $EPOCHS_A" "c $REPS_C $BATCH_C $EPOCHS_C" "d $REPS_D $BATCH_D $EPOCHS_D"; do
  set -- $spec
  arm=$1; reps=$2; bs=$3; eps=$4
  r=0
  while [ "$r" -lt "$reps" ]; do
    if ! run_one "$arm" "$r" "$bs" "$eps"; then
      rc=$?
      if [ "$rc" = "9" ]; then stopped=1; break; fi
    fi
    r=$((r + 1))
  done
  [ "$stopped" = "1" ] && break
done

if [ "$stopped" = "1" ]; then
  say "=== EXP1 GRID STOPPED EARLY $(date) (disk or stop_at) ==="
else
  say "=== EXP1 GRID COMPLETE $(date) ==="
fi
$PY "$PR/exp1_analyse.py" --result "$RES" 2>&1 | tee -a "$GRIDLOG" || true
harvest
say "=== EXP1 DONE $(date) ==="
