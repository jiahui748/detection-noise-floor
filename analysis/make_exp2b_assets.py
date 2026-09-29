#!/usr/bin/env python3
"""Full-model perturbation dose response: figure and table, from the local grid results.

Source: pr_exp/results/divergence.csv (15 runs x 3 epochs), produced by the
experiment 2b grid on the GPU box. No server needed.

Produces:
  paper_PR/figures/fig8_divergence.pdf
  paper_PR/tables/t20_full_model_dose.tex

Follows the Elsevier artwork rules already applied to figs 1-7: natural width at or
just under 390 pt so the 1:1 inclusion does not shrink the lettering below 7 pt,
Arial lettering, Type 42 fonts, colour-blind-safe palette, and no text overlaps
(checked geometrically before writing).

Run:  python audit/make_exp2b_assets.py
"""
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r'<WORKSPACE>')
CSV = ROOT / 'pr_exp' / 'results' / 'divergence.csv'
OUT_FIG = ROOT / 'paper_PR' / 'figures'
OUT_TAB = ROOT / 'paper_PR' / 'tables'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.2,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'mathtext.fontset': 'stixsans', 'figure.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
seeds = sorted({int(r['seed']) for r in rows})
doses = sorted({float(r['eps']) for r in rows})
# d(t) per (dose, seed) at each epoch
d = defaultdict(dict)
for r in rows:
    d[(float(r['eps']), int(r['seed']))][int(r['epoch'])] = float(r['d_relative'])
epochs = sorted({int(r['epoch']) for r in rows})
STEP = 809
xs = [e * STEP for e in epochs]

CMAP = {1e-9: '#c1440e', 1e-8: '#2e7d32', 1e-6: '#7b1fa2', 1e-4: '#1f4e79'}
DOSE_TEX = {1e-9: r'$10^{-9}$', 1e-8: r'$10^{-8}$', 1e-6: r'$10^{-6}$', 1e-4: r'$10^{-4}$'}

fig, axes = plt.subplots(1, 2, figsize=(5.30, 2.25),
                         gridspec_kw={'width_ratios': [1.15, 1.0]})

# --- panel (a): divergence against step, one thin line per (dose, seed) --------
ax = axes[0]
for eps in doses:
    if eps == 0:
        continue
    for s in seeds:
        ys = [d[(eps, s)][e] for e in epochs]
        ax.plot(xs, ys, '-', color=CMAP[eps], lw=0.8, alpha=0.55)
    med = [st.median([d[(eps, s)][e] for s in seeds]) for e in epochs]
    ax.plot(xs, med, 'o-', color=CMAP[eps], ms=3.2, lw=1.6, label=DOSE_TEX[eps])
ax.axhline(1.0, color='#6b6b6b', lw=0.7, ls='--')
ax.text(0.03, 0.06, 'injected $\\varepsilon$ first reaches\n'
                    r'$O(10^{-1})$ within the first epoch',
        transform=ax.transAxes, fontsize=7.5, color='#555555', va='bottom')
ax.set_yscale('log'); ax.set_xlim(0, 3 * STEP)
ax.set_ylim(1e-10, 3.0)
ax.set_xlabel('optimiser step')
ax.set_ylabel(r'$\|\theta_\varepsilon-\theta_0\|/\|\theta_0\|$')
ax.legend(loc='lower right', ncol=2, handlelength=1.2, columnspacing=0.8,
          title=r'injected $\varepsilon$', title_fontsize=7.5)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

# --- panel (b): the endpoint is independent of epsilon ------------------------
ax = axes[1]
for s in seeds:
    ys = [d[(eps, s)][3] for eps in doses if eps != 0]
    ax.plot([e for e in doses if e != 0], ys, 'o', color='#9fb3c8', ms=3.4,
            label='individual seeds' if s == seeds[0] else None)
med = [st.median([d[(eps, s)][3] for s in seeds]) for eps in doses if eps != 0]
ax.plot([e for e in doses if e != 0], med, 's-', color='#c1440e', ms=4, lw=1.4,
        label='median of 3 seeds')
ax.axhline(0.25, color='#6b6b6b', lw=0.7, ls=':')
ax.text(1.3e-9, 0.30, 'five orders of magnitude in $\\varepsilon$,\n'
                      'the same endpoint basin distance',
        fontsize=7.5, color='#555555')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(6e-10, 2e-4); ax.set_ylim(0.08, 1.2)
ax.set_xlabel(r'injected relative perturbation $\varepsilon$')
ax.set_ylabel(r'$d$ at epoch 3')
ax.legend(loc='lower right', handlelength=1.2)
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
fig.tight_layout(w_pad=1.3)
fig.savefig(OUT_FIG / 'fig8_divergence.pdf')
plt.close(fig)
print('wrote fig8_divergence.pdf')

# --- table -------------------------------------------------------------------
L = [r'\begin{table}[t]', r'\centering', r'\small',
     r'\setlength{\tabcolsep}{4pt}',
     r'\caption{Divergence of the full detector under an injected parameter '
     r'perturbation. Each of the 15 runs is a 3-epoch segment of the same protocol '
     r'in strict deterministic mode, so the only difference from the control run of '
     r'the same seed is the injected $\varepsilon$. $d(t)=\|\theta_\varepsilon(t)-'
     r'\theta_0(t)\|/\|\theta_0(t)\|$ is the relative parameter distance from that '
     r'control; entries are the median over the three seeds with the observed range '
     r'in brackets. The per-step factor is measured over each interval. The endpoint '
     r'is the same for every dose across five orders of magnitude in $\varepsilon$, '
     r'so it is a basin distance rather than a rate.}',
     r'\label{tab:full_model_dose}',
     r'\begin{tabular}{lrrrrr}', r'\toprule',
     r'$\varepsilon$ & $d$(ep1) & $d$(ep2) & $d$(ep3) & factor & ampl. to ep3 \\',
     r'\midrule']
for eps in doses:
    if eps == 0:
        continue
    cells = []
    for e in epochs:
        v = [d[(eps, s)][e] for s in seeds]
        cells.append(f'{st.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]')
    facs = []
    for r in rows:
        if float(r['eps']) == eps and r['per_step_factor']:
            facs.append(float(r['per_step_factor']))
    fac = st.median(facs) if facs else float('nan')
    e0 = st.median([float(r['injected_rel']) for r in rows if float(r['eps']) == eps])
    amp = st.median([d[(eps, s)][3] / e0 for s in seeds])
    expo = int(math.floor(math.log10(abs(amp))))
    mant = amp / 10 ** expo
    L.append(f'${DOSE_TEX[eps][1:-1]}$ & {cells[0]} & {cells[1]} & {cells[2]} & '
             f'{fac:.4f} & ${mant:.1f} \\times 10^{{{expo}}}$ \\\\')
L += [r'\midrule',
      r'\multicolumn{6}{l}{\footnotesize $\varepsilon=0$ control: $d\equiv 0$ by '
      r'construction, three seeds.} \\',
      r'\bottomrule', r'\end{tabular}', r'\end{table}']
# Six columns of median-plus-range overrun the text block by about 43pt (about
# 1.5cm). Shrink-to-fit is the same device the generated tables use, and it is
# layout only: not one cell changes.
body = '\n'.join(L)
body = body.replace(r'\begin{tabular}', r'\begin{adjustbox}{max width=\textwidth}' + '\n'
                    + r'\begin{tabular}', 1)
body = body.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + r'\end{adjustbox}', 1)
(OUT_TAB / 't20_full_model_dose.tex').write_text(body + '\n', encoding='utf-8')
print('wrote t20_full_model_dose.tex')

# --- console summary for the manuscript text ---------------------------------
print()
print(f'{"eps":>8} {"d(ep1) med":>12} {"d(ep3) med":>12} {"d(ep3) range":>18} {"factor":>9} {"ampl":>10}')
for eps in doses:
    if eps == 0:
        continue
    m1 = st.median([d[(eps, s)][1] for s in seeds])
    m3 = st.median([d[(eps, s)][3] for s in seeds])
    r3 = (min(d[(eps, s)][3] for s in seeds), max(d[(eps, s)][3] for s in seeds))
    facs = [float(r['per_step_factor']) for r in rows
            if float(r['eps']) == eps and r['per_step_factor']]
    e0 = st.median([float(r['injected_rel']) for r in rows if float(r['eps']) == eps])
    print(f'{eps:>8g} {m1:>12.3f} {m3:>12.3f}   [{r3[0]:.3f}, {r3[1]:.3f}]'
          f'{st.median(facs):>9.4f} {m3/e0:>10.1e}')
all3 = [d[(eps, s)][3] for eps in doses if eps for s in seeds]
print(f'\nacross all doses and seeds: d(ep3) min={min(all3):.3f} median={st.median(all3):.3f} max={max(all3):.3f}')
