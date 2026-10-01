cd <REMOTE-ROOT> || exit 1
export PYTHONPATH=<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main:<REMOTE-ROOT>/mw:<REMOTE-ROOT>
PY=<REMOTE-ROOT>/envs/mw/bin/python
L=<REMOTE-ROOT>/pr_exp/logs
mkdir -p "$L" <REMOTE-ROOT>/mw/pr_exp/snap_intra <REMOTE-ROOT>/mw/pr_exp/result_intra
: > "$L/intra.log"

for eps in 0 1e-9 1e-8; do
  name="intra_e${eps}_s0"
  echo "--- $name START $(date) ---" >> "$L/intra.log"
  t0=$(date +%s)
  $PY -u <REMOTE-ROOT>/pr_exp/exp2c_intra.py --eps "$eps" --seed 0 --epochs 1 \
      --every 100 --name "$name" >> "$L/intra.log" 2>&1
  echo "--- $name EXIT rc=$? wall=$(( ($(date +%s)-t0)/60 ))min ---" >> "$L/intra.log"
done
echo "=== INTRA COMPLETE $(date) ===" >> "$L/intra.log"
