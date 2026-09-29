cd /root/autodl-tmp || exit 1
export PYTHONPATH=/root/autodl-tmp/mw/RTDETR-main/RTDETR-main:/root/autodl-tmp/mw:/root/autodl-tmp
PY=/root/autodl-tmp/envs/mw/bin/python
L=/root/autodl-tmp/pr_exp/logs

# wait for the intra-epoch sweep to finish before starting 2a, so the two never
# contend for the ~24 GB the trainer needs
while ! grep -q 'INTRA COMPLETE' "$L/intra.log" 2>/dev/null; do sleep 20; done
echo "intra done at $(date), starting 2a" >> "$L/realact.log"

$PY -u /root/autodl-tmp/pr_exp/exp2a_realact.py --batch 8 --reps 8 \
    >> "$L/realact.log" 2>&1
echo "REALACT EXIT rc=$? at $(date)" >> "$L/realact.log"
