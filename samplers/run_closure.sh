
set -u
ROOT=<REMOTE-ROOT>
MW=$ROOT/mw
PY=$ROOT/envs/mw/bin/python
LOG=$ROOT/logs/closure.log
: > "$LOG"
cd $MW || exit 1
export PYTHONPATH=$MW/RTDETR-main/RTDETR-main:$ROOT

for p in $ROOT/phase8_train_v2.py $ROOT/phase8_hook.py $MW/det_deform_attn.py \
         $MW/visdrone_size_probe.py $MW/VisDrone/VisDrone.yaml; do
  [ -e "$p" ] || { echo "PREFLIGHT FAIL: missing $p"; exit 3; }
done
echo "preflight OK" | tee -a "$LOG"

for i in 1 2 3; do
  name="closure_s1e8_r$i"
  W=$MW/result_sdd/$name/weights/best.pt
  P=$MW/probe_out/size_probe_${name}_val.json
  if [ -f "$P" ]; then echo "SKIP $name (probed)" | tee -a "$LOG"; continue; fi
  t0=$(date +%s)
  $PY -u $ROOT/phase8_train_v2.py --sigma 1e-8 --seed 0 --epochs 30 \
      --batch 8 --workers 8 --imgsz 640 --name "$name" \
      --project $MW/result_sdd >> "$LOG" 2>&1
  rc=$?
  echo "EXIT $name rc=$rc wall=$(( ($(date +%s)-t0)/60 ))min" | tee -a "$LOG"
  if [ -f "$W" ]; then
    find $MW/result_sdd/$name/weights -name "epoch*.pt" -delete 2>/dev/null
    timeout 3600 $PY -u $MW/visdrone_size_probe.py "$W" --imgsz 640 --conf 0.05 \
        --split val --tag "${name}_val" >> "$LOG" 2>&1
    echo "PROBED $name" | tee -a "$LOG"
  fi
done
echo "CLOSURE COMPLETE" | tee -a "$LOG"
