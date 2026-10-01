cd <REMOTE-ROOT> || exit 1
export PYTHONPATH=<REMOTE-ROOT>/mw/RTDETR-main/RTDETR-main:<REMOTE-ROOT>/mw:<REMOTE-ROOT>
PY=<REMOTE-ROOT>/envs/mw/bin/python
L=<REMOTE-ROOT>/pr_exp/logs

# wait for the intra-epoch sweep to finish before starting 2a, so the two never
# contend for the ~24 GB the trainer needs
while ! grep -q 'INTRA COMPLETE' "$L/intra.log" 2>/dev/null; do sleep 20; done
echo "intra done at $(date), starting 2a" >> "$L/realact.log"

$PY -u <REMOTE-ROOT>/pr_exp/exp2a_realact.py --batch 8 --reps 8 \
    >> "$L/realact.log" 2>&1
echo "REALACT EXIT rc=$? at $(date)" >> "$L/realact.log"
