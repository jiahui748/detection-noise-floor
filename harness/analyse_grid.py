#!/usr/bin/env python3
"""Analyse the experiment 2b dose grid.

For each seed, the eps=0 run is the control. For every dose the relative parameter
divergence from that control is computed at each snapshot epoch,

    d(t) = ||theta_eps(t) - theta_0(t)|| / ||theta_0(t)||

and the per-step amplification factor over each interval is

    factor([t1,t2]) = (d(t2) / d(t1)) ** (1 / (t2 - t1))

with d(0) = the injected epsilon, so the first interval is measured from the
injection rather than from a snapshot. This is the full-model analogue of Table 11,
whose per-step factors (1.007-1.029) came from a standalone module.

The final mAP50 of each run is parsed out of the grid log so the metric spread across
doses and seeds can be compared with the 0.3-point mAP50 standard deviation the paper
reports for the real reference arm.

Run on the box:  python analyse_grid.py
"""
import glob
import json
import math
import os
import re
import sys

SNAP = '<REMOTE-ROOT>/pr_exp/snap'
LOG = '<REMOTE-ROOT>/pr_exp/logs/grid.log'
ITERS_PER_EPOCH = 809


def load_flat(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    m = sd.get('model', sd)
    if hasattr(m, 'state_dict'):
        m = m.state_dict()
    vec = []
    for k in sorted(m):
        v = m[k]
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            vec.append(v.detach().reshape(-1))
    import numpy as np
    return np.concatenate([t.numpy().astype('float64') for t in vec])


def run_dir(name):
    return os.path.join(SNAP, name)


def meta(name):
    p = os.path.join(run_dir(name), 'meta.json')
    return json.load(open(p)) if os.path.exists(p) else None


def epochs_available(name):
    return sorted(int(re.search(r'ep(\d+)\.pt', f).group(1))
                  for f in glob.glob(os.path.join(run_dir(name), 'ep*.pt')))


def parse_map50_from_log():
    """Final 'all  <imgs> <inst> P R mAP50 mAP50-95' line following each run START."""
    out = {}
    if not os.path.exists(LOG):
        return out
    cur = None
    with open(LOG, errors='replace') as f:
        for line in f:
            line = line.replace('\r', '\n')
            for piece in line.split('\n'):
                m = re.search(r'--- (traj_\S+) START', piece)
                if m:
                    cur = m.group(1)
                    continue
                m = re.match(r'\s*all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s*$',
                             piece.strip())
                if m and cur:
                    out[cur] = float(m.group(1))
    return out


def main():
    import numpy as np
    names = sorted(os.path.basename(d) for d in glob.glob(f'{SNAP}/traj_*'))
    if not names:
        print('no runs found yet')
        return
    map50 = parse_map50_from_log()
    seeds = sorted({int(n.split('_s')[-1]) for n in names})
    doses = sorted({float(n.split('_')[1][1:]) for n in names if re.match(r'traj_e[0-9e.-]+_s\d+', n)})
    print(f'runs found: {len(names)}   seeds: {seeds}   doses: {doses}')
    print()

    report = {}
    for s in seeds:
        ctrl = f'traj_e0_s{s}'
        if not os.path.exists(os.path.join(run_dir(ctrl), 'meta.json')):
            print(f'seed {s}: control {ctrl} not finished, skipping')
            continue
        eps_ctrl = epochs_available(ctrl)
        base = {e: load_flat(os.path.join(run_dir(ctrl), f'ep{e}.pt')) for e in eps_ctrl}
        roll = {}
        for eps in doses:
            name = f'traj_e{eps:g}_s{s}' if eps else f'traj_e0_s{s}'
            if not os.path.exists(os.path.join(run_dir(name), 'meta.json')):
                continue
            mt = meta(name)
            eps_inj = mt['injected_rel'] if eps else 0.0
            row = []
            prev_d, prev_t = (eps_inj if eps_inj > 0 else None), 0
            for e in epochs_available(name):
                t = e * ITERS_PER_EPOCH
                if eps == 0:
                    row.append((t, 0.0, 0.0))
                    continue
                th = load_flat(os.path.join(run_dir(name), f'ep{e}.pt'))
                d = float(np.linalg.norm(th - base[e]) / max(np.linalg.norm(base[e]), 1e-30))
                fac = float('nan')
                if prev_d and prev_d > 0 and t > prev_t:
                    fac = (d / prev_d) ** (1.0 / (t - prev_t))
                row.append((t, d, fac))
                prev_d, prev_t = d, t
            roll[eps] = row
            print(f'seed {s}  eps={eps:g}  injected={eps_inj:.3e}')
            for t, d, fac in row:
                f = f'{fac:.4f}' if fac == fac else '   --  '
                print(f'    t={t:>5}  d(t)={d:.6e}   per-step={f}')
            if name in map50:
                print(f'    final mAP50 = {map50[name]:.4f}')
        report[s] = roll

    print()
    print('=== metric spread across doses, per seed ===')
    for s in seeds:
        vals = []
        for eps in doses:
            name = f'traj_e{eps:g}_s{s}' if eps else f'traj_e0_s{s}'
            if name in map50:
                vals.append(map50[name])
        if len(vals) > 1:
            import statistics as st
            print(f'  seed {s}: n={len(vals)} mean={st.mean(vals):.4f} '
                  f'sd={st.stdev(vals):.5f} range={max(vals)-min(vals):.5f}')
    json.dump({str(k): {str(e): v for e, v in r.items()} for k, r in report.items()},
              open('<REMOTE-ROOT>/pr_exp/divergence.json', 'w'), indent=1)
    print('\nwrote <REMOTE-ROOT>/pr_exp/divergence.json')


if __name__ == '__main__':
    sys.exit(main())
