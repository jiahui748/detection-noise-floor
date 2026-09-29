cd /root/autodl-tmp || exit 1
export PYTHONPATH=/root/autodl-tmp/mw/RTDETR-main/RTDETR-main:/root/autodl-tmp/mw:/root/autodl-tmp
PY=/root/autodl-tmp/envs/mw/bin/python
L=/root/autodl-tmp/pr_exp/logs
mkdir -p "$L" /root/autodl-tmp/mw/pr_exp/snap_intra /root/autodl-tmp/mw/pr_exp/result_intra
: > "$L/intra.log"

for eps in 0 1e-9 1e-8; do
  name="intra_e${eps}_s0"
  echo "--- $name START $(date) ---" >> "$L/intra.log"
  t0=$(date +%s)
  $PY -u /root/autodl-tmp/pr_exp/exp2c_intra.py --eps "$eps" --seed 0 --epochs 1 \
      --every 100 --name "$name" >> "$L/intra.log" 2>&1
  echo "--- $name EXIT rc=$? wall=$(( ($(date +%s)-t0)/60 ))min ---" >> "$L/intra.log"
done
echo "=== INTRA COMPLETE $(date) ===" >> "$L/intra.log"
