#!/usr/bin/env python3
"""Regenerate all seven figures of the manuscript, to Elsevier artwork rules.

The rules this file is written against (Elsevier "Artwork and media instructions",
Artwork sizing / Artwork types / Formats checklist):

  * lettering must have a **finished printed size of 7 pt** for normal text and no
    smaller than 6 pt for sub/superscripts; "smaller lettering will yield text that
    is hardly legible";
  * line weights should be within **0.10-1.5 pt**;
  * vector artwork should be **EPS or PDF**, fonts **embedded**, preferably
    Arial/Helvetica/Courier/Times/Symbol;
  * RGB colourspace, colour-blind-safe palette, sufficient contrast;
  * "no data should be present outside the actual illustration area";
  * physical dimensions should match the journal's column widths
    (single 90 mm / 1.5-column 140 mm / double 190 mm).

What was wrong before, and what this file fixes:

  1. Every figure was included at `width=\\textwidth` = 390 pt in the elsarticle
     preprint. Figures 3-7 were authored 6.9 in (497 pt) wide, so they were scaled
     DOWN by 0.79 and their 8 pt lettering printed at ~6.3 pt --- below the 7 pt
     rule. Figure 2 was authored 3.4 in (245 pt) wide and scaled UP by 1.6 to a
     304 pt-tall panel. Both are now authored so that the natural width is at or
     just under 390 pt, so the scale factor is ~1.0 and the lettering prints at the
     size it is set in.
  2. matplotlib's default PDF backend emitted **Type 3** fonts. `pdf.fonttype = 42`
     forces TrueType (Type 42) fonts, which are embedded, scalable and text-
     extractable, as Elsevier production requires.
  3. Text was set in DejaVu Serif, which is not on Elsevier's recommended list.
     Figures now use **Arial** (Helvetica), a recommended font, which is also the
     usual convention for figure lettering.
  4. Fig. 1 carried far too much text in 5.2-7.0 pt type. It is redrawn in point
     coordinates with four boxes per row and no label below 7.5 pt; the detail that
     was crammed into the boxes now lives in the caption, where a reader can
     actually read it.

Run:  python audit/make_figures.py
"""
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / 'remote_json'
VIS = R / 'mw' / 'probe_out'
TT = R / 'tt100k' / 'probe_out'
FIG = R / 'figdata'
EPOCHS = R / 'epochs'
OUT = ROOT.parent / 'paper_PR' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

TEXTWIDTH_PT = 390.0          # elsarticle preprint \textwidth, from the build log
FULL = 5.30                   # inches: a two-panel figure at ~1:1
LEGAL_MIN = 7.0               # Elsevier: finished lettering size for normal text

plt.rcParams.update({
    # --- Elsevier artwork rules -------------------------------------------
    'pdf.fonttype': 42,               # TrueType, not Type 3
    'ps.fonttype': 42,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    # --- shared style ------------------------------------------------------
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    # STIX sans for math: sans so it matches Arial, and STIX is the Symbol-family
    # lineage Elsevier lists among its recommended figure fonts.
    'mathtext.fontset': 'stixsans',
})

C1 = '#1f4e79'   # dark blue
C2 = '#c1440e'   # burnt orange
C3 = '#2e7d32'   # green
C4 = '#7b1fa2'   # purple
GREY = '#555555'

METRICS = ['mAP50', 'mAP50-95', '<4px', '4-8px', '8-16px', '16-32px', '32-64px', '>64px']
LABEL = {'mAP50': r'mAP$_{50}$', 'mAP50-95': r'mAP$_{50\text{-}95}$'}
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)

# The perturbation sweep. The JSON keys are strings like "1e-10" and "0.01"; the
# numeric epsilon must be stated separately. Writing `10 ** float(key)` looks right
# and is wrong: float("1e-10") is already the value 1e-10, so 10**1e-10 is 1.0000000002
# and every point lands at x = 1 on a log axis.
EPS_KEYS = ['1e-10', '1e-08', '1e-06', '0.0001', '0.01']
EPS_VAL = {'1e-10': 1e-10, '1e-08': 1e-8, '1e-06': 1e-6, '0.0001': 1e-4, '0.01': 1e-2}
EPS_TEX = {'1e-10': r'$10^{-10}$', '1e-08': r'$10^{-8}$', '1e-06': r'$10^{-6}$',
           '0.0001': r'$10^{-4}$', '0.01': r'$10^{-2}$'}

REF10 = [f'pn_base_s{i}' for i in range(10)]
PHASE7 = [f'orig_s{i}' for i in range(3, 8)]
REF15 = REF10 + PHASE7
SEED0 = ['pn_base_s0', 'pn_base_s0_rep'] + [f'pn_base_s0_j{i}' for i in range(2, 6)]
DET_ONLY = ['det_base_s0_d1', 'det_base_s0_d2', 'det_base_s0_d3']
SAMP_ONLY3 = ['detsamponly_s0_r1', 'samponly_s0_r2', 'samponly_s0_r3']
SAMP_STRICT = ['detsamp_s0_r1', 'detsamp_s0_r2', 'detsamp_s0_r3']
EARLY = REF10 + PHASE7 + SEED0[1:] + SAMP_ONLY3 + [f'detsamp_s{i}' for i in range(1, 6)] + SAMP_STRICT
BAD = 'detsamp_s2'


def load(name, base=VIS):
    d = json.loads((base / f'size_probe_{name}_val.json').read_text())
    out = {k: d['overall'][k] for k in ('mAP50', 'mAP50-95')}
    for k, v in d['by_size'].items():
        out[k] = v.get('ap50')
    return out


def arm(names, base=VIS):
    return {n: load(n, base) for n in names}


def col(rows, m):
    return [r[m] for r in rows.values() if r.get(m) is not None]


def overlaps(fig, name):
    """Report pairs of Text artists, within the same axes, whose boxes intersect.

    Eyeballing a rendered PNG misses collisions that only appear at final print
    size, so every figure is checked geometrically before it is written. Text in
    different panels cannot collide visually unless the panels themselves overlap
    (a separate layout error), so only same-axes pairs are compared. A 1.5 px
    tolerance avoids flagging text that merely touches.

    Known false-positive source: matplotlib sometimes keeps an out-of-range tick
    label as a visible artist that is never painted (e.g. "0.98" on an axis whose
    limits start at 0.9845). Those are reported here but do not exist in the PDF;
    `pdftotext -bbox` on the written figure is the arbiter. Figures 5 and 7 each
    carry one such flag, checked and confirmed absent in the output.
    """
    import matplotlib.text as mtext
    import matplotlib.transforms as mtransforms
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            box = t.get_window_extent(renderer=r)
        except Exception:
            continue
        items.append((t.get_text()[:24].replace('\n', '/'), box, id(t.axes)))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if items[i][2] is not None and items[j][2] is not None \
                    and items[i][2] != items[j][2]:
                continue                      # different panels: cannot collide
            inter = mtransforms.Bbox.intersection(a, b)
            if inter is None:
                continue
            # severity: a marginal touch between adjacent tick labels is harmless, but
            # a caption running into the next column is not. 10% of the smaller
            # label's area catches the latter and ignores the former.
            small = min(a.width * a.height, b.width * b.height)
            if small > 0 and (inter.width * inter.height) / small > 0.10:
                bad.append((items[i][0], items[j][0],
                            round(inter.width, 1), round(inter.height, 1)))
    if bad:
        print(f'  !! {name}: {len(bad)} text overlap(s):')
        for a, b, w, h in bad[:10]:
            print(f'       "{a}" x "{b}"  ({w}x{h} px)')
    else:
        print(f'  ok {name}: no text overlaps')
    return bad


def save(fig, name, tight=True):
    overlaps(fig, name)
    p = OUT / f'{name}.pdf'
    fig.savefig(p, bbox_inches='tight' if tight else None)
    plt.close(fig)
    print('wrote', p.name)


# ===========================================================================
# Fig 1 - visual abstract.
#
# Replaces the original boxes-and-arrows flow chart. A flow chart of labelled
# rectangles carries no evidence: every panel here instead plots the paper's own
# measurements, so the overview figure is itself a result. Four stages read left
# to right: the practice, the floor it ignores, the mechanism, the remedy.
#
# The figure is authored at 390 x 156 pt = 2.5:1 with every label >= 7.5 pt, which
# satisfies two constraints at once:
#   * as Fig. 1 of the paper it is included at width=\textwidth (390 pt), so it is
#     neither shrunk nor enlarged and the lettering prints at the size it is set;
#   * as a graphical abstract it meets Elsevier's rule of a minimum of
#     1328 x 531 px at 300 dpi in a 2.5:1 ratio (500 x 200 px window).
# See audit/make_figures.py header for the artwork rules and their source.
# ===========================================================================
def fig1_visual_abstract():
    ref = arm(REF15)
    mde48 = st.stdev(col(ref, '4-8px')) * Z * math.sqrt(2) * 100.0   # AP points
    sig_agg = st.stdev(col(ref, 'mAP50'))
    bin_sig = [st.stdev(col(ref, m)) for m in METRICS[2:]]

    W, H = TEXTWIDTH_PT, 156.0                     # 2.5 : 1
    M, GAP = 3.0, 10.0
    CW = (W - 2 * M - 3 * GAP) / 4.0
    PY0, PH = 62.0, 76.0                           # plot band: 62 .. 138
    HEAD_Y, CAP_Y = 149.0, 53.0

    fig = plt.figure(figsize=(W / 72, H / 72))
    ink, soft, line = '#1a1a1a', '#6b6b6b', '#dcdcdc'
    ACC, WARN = C1, C2

    def fx(pt):
        return pt / W

    def fy(pt):
        return pt / H

    def col_x(i):
        return M + i * (CW + GAP)

    def add(i, y0, h):
        return fig.add_axes([fx(col_x(i)), fy(y0), CW / W, h / H])

    def bare(ax):
        """Strip everything that is not data.

        `set_xticks([])` clears only the major locator, so log-scale minor ticks
        (and their labels) survive it; the locators must be replaced outright.
        """
        from matplotlib.ticker import NullLocator
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(NullLocator())
            axis.set_minor_locator(NullLocator())
        ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        for s in ax.spines.values():
            s.set_visible(False)

    # stage headers and hairline separators
    for i, lab in enumerate(('1  the practice', '2  the noise floor',
                             '3  the mechanism', '4  the remedy')):
        fig.text(fx(col_x(i) + CW / 2), fy(HEAD_Y), lab, ha='center', va='center',
                 fontsize=8.2, fontweight='bold', color=ACC if i != 3 else C3)
    for i in (1, 2, 3):
        fig.add_artist(plt.Line2D([fx(col_x(i) - GAP / 2)] * 2, [fy(8), fy(H - 16)],
                                  color=line, lw=0.6, transform=fig.transFigure))

    # -- stage 1: a typical claim against the floor it has to clear -----------
    ax = add(0, PY0, PH)
    vals = [2.76, mde48]
    ax.bar([0.0, 1.0], vals, width=0.6, color=[ACC, WARN], zorder=3)
    ax.text(0.0, vals[0] + 0.35, 'survey median\n+2.8', ha='center', va='bottom',
            fontsize=7.5, color=ink, linespacing=1.3)
    ax.text(1.0, vals[1] + 0.35, '1-run MDE\n$\\pm$10.8', ha='center', va='bottom',
            fontsize=7.5, color=ink, linespacing=1.3)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 17.5)
    bare(ax)

    # -- stage 2: the per-bin floor, which is what the claim is measured on --
    ax = add(1, PY0, PH)
    xi = np.arange(len(METRICS[2:]))
    peak = {'4-8px', '>64px'}
    ax.bar(xi, bin_sig, width=0.62,
           color=[WARN if m in peak else '#9fb3c8' for m in METRICS[2:]], zorder=3)
    ax.axhline(sig_agg, color=soft, lw=0.8, ls='--', zorder=2)
    ax.annotate('$9.9\\times$', xy=(1.0, bin_sig[1]), xytext=(1.0, bin_sig[1] * 1.25),
                ha='center', va='bottom', fontsize=7.5, color=ink)
    # no in-plot label for the dashed line: at this column width it lands on the
    # bars. The caption names it instead.
    ax.set_yscale('log'); ax.set_ylim(2.0e-3, 0.14); ax.set_xlim(-0.7, 5.7)
    bare(ax)

    # -- stage 3: two sparklines, the measured causal chain ------------------
    gn = json.loads((FIG / 'gradnoise.json').read_text())
    tg = json.loads((FIG / 'theory_growth.json').read_text())
    axt = add(2, 102.0, 36.0)
    axt.loglog([r['points'] for r in gn['C']], [r['grid_sd'] for r in gn['C']],
               'o-', color=WARN, ms=2.8, lw=1.0)
    # labels go top-left, where both curves leave the corner empty; a label long
    # enough to be readable would otherwise run under the rising line
    axt.text(0.03, 0.94, '$279\\times$ growth', transform=axt.transAxes,
             ha='left', va='top', fontsize=7.5, color=ink)
    axt.set_xlim(5e2, 8e4); axt.set_ylim(1e-10, 4e-6)
    bare(axt)
    axb = add(2, PY0, 36.0)
    axb.semilogx([EPS_VAL[k] for k in EPS_KEYS], [tg['per_step'][k] for k in EPS_KEYS],
                 'o-', color=ACC, ms=2.8, lw=1.2)
    axb.axhline(1.0, color=soft, lw=0.8, ls='--')
    axb.text(0.97, 0.94, 'gain 1.007--1.029', transform=axb.transAxes,
             ha='right', va='top', fontsize=7.5, color=ink)
    axb.set_xlim(3e-11, 3e-1); axb.set_ylim(0.978, 1.055)
    bare(axb)

    # -- stage 4: sigma against wall-clock, the trade the reader must make ---
    ax = add(3, PY0, PH)
    pts = [(1.00, st.stdev(col(arm(SEED0), 'mAP50')), '#9fb3c8'),
           (1.83, st.stdev(col(arm(DET_ONLY), 'mAP50')), '#9fb3c8'),
           (1.03, st.stdev(col(arm(SAMP_ONLY3), 'mAP50')), WARN),
           (1.78, 4.0e-4, C3)]
    for x, y, c in pts:
        ax.plot([x], [y], 'o', ms=5.0, color=c, zorder=4, clip_on=False)
    ax.annotate('$\\sigma=0$ (at the floor)', xy=(1.78, 4.0e-4), xytext=(1.70, 1.15e-3),
                ha='right', va='bottom', fontsize=7.5, color=C3,
                arrowprops=dict(arrowstyle='-|>', lw=0.6, color=C3, shrinkA=1, shrinkB=4))
    ax.set_yscale('log'); ax.set_xlim(0.88, 2.02); ax.set_ylim(2.3e-4, 9e-3)
    ax.text(1.00, 2.62e-4, '1.0$\\times$', ha='center', va='bottom', fontsize=7.5, color=soft)
    # 1.9 rather than 1.83: the sampler+strict marker sits at 1.78 and the two
    # labels would otherwise be 4 pt apart
    ax.text(1.92, 2.62e-4, '1.8$\\times$', ha='center', va='bottom', fontsize=7.5, color=soft)
    ax.text(0.90, 7.2e-3, 'run-to-run $\\sigma$', ha='left', va='top', fontsize=7.5,
            color=soft)
    bare(ax)

    # -- one-line reading under each stage ----------------------------------
    # each line is kept under ~22 characters, which is all an 88 pt column takes
    caps = [
        'A one-run claim is\nsmaller than the\nfloor it must clear.',
        'Dashed line: the\naggregate floor. Peaks\nat $9.9\\times$, in between.',
        'Order-dependent\nnoise, amplified\nby the optimiser.',
        'Only both together\nmake runs\nbit-identical; cost\n1.0 -- 1.8$\\times$.',
    ]
    for i, c in enumerate(caps):
        fig.text(fx(col_x(i)), fy(CAP_Y), c, ha='left', va='top', fontsize=7.5,
                 color=soft, linespacing=1.5)

    overlaps(fig, 'fig1_overview')
    # Save at the exact figure box rather than with the global tight crop: the
    # graphical abstract must be 2.5:1 (Elsevier: "if you are submitting a larger
    # image, please use the same ratio (500 wide x 200 high)"), and tight-cropping
    # trims a few points off the height. At 390 x 156 pt this is exactly 2.5:1, and
    # at 300 dpi it is 1625 x 650 px, above the 1328 x 531 px minimum.
    from matplotlib.transforms import Bbox
    box = Bbox([[0, 0], [W / 72, H / 72]])
    fig.savefig(OUT / 'fig1_overview.pdf', bbox_inches=box)
    fig.savefig(OUT / 'graphical_abstract.pdf', bbox_inches=box)
    fig.savefig(OUT / 'graphical_abstract.png', dpi=300, bbox_inches=box)
    try:
        fig.savefig(OUT / 'graphical_abstract.tif', dpi=300, bbox_inches=box,
                    pil_kwargs={'compression': 'tiff_lzw'})
    except Exception as exc:                                   # pragma: no cover
        print(f'  (tif not written: {type(exc).__name__}: {exc})')
    plt.close(fig)
    print('wrote fig1_overview.pdf + graphical_abstract.{pdf,png,tif} (2.5:1)')


fig1_visual_abstract()

# ===========================================================================
# Fig 2 - runs per arm (single panel; included at 0.66\textwidth, so authored
# at ~258 pt rather than stretched to 390 pt)
# ===========================================================================
ref15 = arm(REF15)
fig, ax = plt.subplots(figsize=(3.62, 2.72))
deltas = np.logspace(math.log10(0.1), math.log10(50), 200)
cmap = plt.get_cmap('viridis')
for i, m in enumerate(METRICS):
    s = st.stdev(col(ref15, m))
    ax.plot(deltas, 2 * s ** 2 * Z ** 2 / (deltas / 100) ** 2,
            color=cmap(i / (len(METRICS) - 1)), label=LABEL.get(m, m))
ax.axhline(1, color=GREY, lw=0.6, ls=':')
ax.axvline(1, color=GREY, lw=0.6, ls=':')
ax.axvspan(0.1, 4.4, color='#000000', alpha=0.05, lw=0)
ax.text(0.66, 1.6e4, 'typical reported\nimprovement ($\\lesssim$4.4 AP)',
        fontsize=7.5, color=GREY, ha='center', va='top')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'true improvement $\Delta$AP (points)')
ax.set_ylabel('runs per arm for 80% power')
ax.set_ylim(0.05, 4e4); ax.set_xlim(0.1, 50)
ax.legend(loc='upper right', ncol=2, handlelength=1.1, columnspacing=0.7, borderpad=0.1)
save(fig, 'fig2_runs_per_arm')

# ===========================================================================
# Fig 3 - scale profile (two panels, authored at ~1:1 to 390 pt)
# ===========================================================================
tt6 = arm([f'tt_base_s{i}' for i in range(6)], TT)
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.25), gridspec_kw={'width_ratios': [1.12, 1.35]})
ax = axes[0]
x = np.arange(len(METRICS[2:]))
sv = [st.stdev(col(ref15, m)) for m in METRICS[2:]]
stt = [st.stdev(col(tt6, m)) if len(col(tt6, m)) > 1 else np.nan for m in METRICS[2:]]
ax.bar(x - 0.19, sv, 0.36, color=C1, label='VisDrone (15 runs)')
ax.bar(x + 0.19, stt, 0.36, color=C2, label='TT100K (6 runs)')
# The two value labels per group are wider than the group at this panel width, so
# they are set vertically; horizontal they overlap the neighbouring group's labels.
for xi, v in zip(x - 0.19, sv):
    # rotation_mode='anchor' is essential: without it the rotated text is aligned in
    # the unrotated frame and extends downwards, straight into the tick labels.
    ax.text(xi, v * 1.18, f'{v:.3f}', rotation=90, rotation_mode='anchor',
            ha='left', va='center', fontsize=7.5, color=C1)
for xi, v in zip(x + 0.19, stt):
    if v == v:
        ax.text(xi, v * 1.18, f'{v:.3f}', rotation=90, rotation_mode='anchor',
                ha='left', va='center', fontsize=7.5, color=C2)
    else:
        # one short line at the foot of the empty slot
        ax.text(xi, 1.95e-3, 'n/a', ha='center', va='center', fontsize=7.5, color=GREY)
ax.set_yscale('log'); ax.set_ylim(1.5e-3, 1.7)
SHORT = {'<4px': '<4', '4-8px': '4-8', '8-16px': '8-16', '16-32px': '16-32',
         '32-64px': '32-64', '>64px': '>64'}
ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in METRICS[2:]], rotation=30, ha='right')
ax.set_xlabel('object size bin (native px)')
ax.set_ylabel(r'run-to-run $\sigma$ (AP$_{50}$)')
# legend above the axes: inside them it collides with the taller bars' value labels
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), handlelength=1.0)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

ax = axes[1]
# Panel (b) labels only the five points that carry the argument --- the two peaks,
# the two calm large-object ends, and the hopelessly small end --- rather than all
# ten. Labelling every point at this panel width produces overlapping text, and the
# full per-bin numbers are already in Table 5 and the caption.
BB = dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.6)
for name, a, c, mk in (('VisDrone', ref15, C1, 'o'), ('TT100K', tt6, C2, 's')):
    xs, ys = [], []
    for m in METRICS[2:]:
        v = col(a, m)
        if len(v) > 1:
            xs.append(st.mean(v)); ys.append(st.stdev(v) / st.mean(v))
    ax.plot(xs, ys, color=c, marker=mk, ms=3.6, lw=0.9, label=name)
KEY = [('VisDrone', '<4px', '<4', (0, 9)), ('VisDrone', '4-8px', '4-8  (peak)', (0, 10)),
       ('TT100K', '8-16px', '8-16  (peak)', (0, 10)), ('VisDrone', '>64px', '>64', (-4, 10)),
       ('TT100K', '>64px', '>64', (0, 11))]
data = {'VisDrone': ref15, 'TT100K': tt6}
for name, m, lab, (dx, dy) in KEY:
    v = col(data[name], m)
    ax.annotate(lab, (st.mean(v), st.stdev(v) / st.mean(v)), textcoords='offset points',
                xytext=(dx, dy), ha='center', va='center', fontsize=7.5,
                color=C1 if name == 'VisDrone' else C2, bbox=BB, zorder=5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.03, 1.5); ax.set_ylim(9e-3, 0.62)
ax.set_xlabel(r'mean AP$_{50}$ of the bin')
ax.set_ylabel(r'coefficient of variation $\sigma/\mu$')
ax.axvspan(0.1, 0.3, color=GREY, alpha=0.07, lw=0)
ax.annotate('partially\nsolved band', xy=(0.185, 0.058), fontsize=7.5, color=GREY,
            ha='center', va='center', bbox=BB)
ax.legend(loc='upper right', handlelength=1.0)
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
fig.tight_layout(w_pad=1.3)
save(fig, 'fig3_scale_profile')

# ===========================================================================
# Fig 4 - gradient noise
# ===========================================================================
gn = json.loads((FIG / 'gradnoise.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
ax = axes[0]
pts = [r['points'] for r in gn['C']]
ax.loglog(pts, [r['grid_sd'] for r in gn['C']], 'o-', color=C2, ms=3.4,
          label='F.grid_sample backward')
ax.loglog(pts, [r['ours_sd'] for r in gn['C']], 's-', color=C1, ms=3.4,
          label='deterministic sampler')
ax.annotate('279$\\times$ from\n900 to 14,400', xy=(2600, 2.0e-7), xytext=(6.5e2, 4.0e-9),
            fontsize=7.5, color=GREY,
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color=GREY, shrinkA=2, shrinkB=4))
ax.annotate('non-monotone\n(different shape)', xy=(40000, 2.97e-7), xytext=(1.6e4, 1.3e-6),
            fontsize=7.5, color=GREY, ha='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color=GREY, shrinkA=2, shrinkB=3))
ax.set_xlabel('number of sampled points')
ax.set_ylabel('sd of repeated\nbackward passes')
ax.set_ylim(6e-11, 4e-6)
ax.legend(loc='lower right', handlelength=1.0)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

ax = axes[1]
# Two-line tick labels: the full "(N,C,H,W)->(Ho x Wo)" string is wider than the slot
# available per bar at this panel width, and the grid dimensions are in the caption.
shapes = [r['shape'] for r in gn['A']]
short = []
for s in shapes:
    lhs = s.split('->')[0].strip('()')
    n, c, h, w = lhs.split(',')
    short.append(f'{n},{c}\n{h},{w}')
idx = np.arange(len(shapes))
ax.bar(idx - 0.19, [r['grid_rel'] for r in gn['A']], 0.36, color=C2, label='F.grid_sample')
vals = [r['ours_rel'] for r in gn['A']]
ax.bar(idx + 0.19, [v if v > 0 else 4e-13 for v in vals], 0.36, color=C1,
       label='deterministic sampler')
for i, v in enumerate(vals):
    if v == 0:
        ax.text(i + 0.19, 5.6e-13, 'exactly\n0', ha='center', fontsize=7.5, color=C1)
ax.set_yscale('log'); ax.set_ylim(2e-13, 9e-8)
ax.set_xticks(idx)
ax.set_xticklabels(short, fontsize=7.5)
ax.set_ylabel('relative gradient noise\n(sd / mean)')
ax.set_xlabel('tensor shape $(N,C$ above, $H,W$ below$)$')
ax.legend(loc='upper left', handlelength=1.0)
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
fig.tight_layout(w_pad=1.3)
save(fig, 'fig4_grad_noise')

# ===========================================================================
# Fig 5 - amplification
# ===========================================================================
tg = json.loads((FIG / 'theory_growth.json').read_text())
rec, dist, per_step = tg['rec'], tg['dist'], tg['per_step']
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
ax = axes[0]
for k, c in zip(EPS_KEYS, [GREY, C2, C3, C4, C1]):
    ax.loglog(np.array(rec) + 1, dist[k], ':' if k == '1e-10' else '-', color=c,
              marker='o', ms=2.6, label=r'$\varepsilon=$' + EPS_TEX[k])
ax.axhline(1.2e-7, color=GREY, lw=0.6, ls='--')
ax.text(1.15, 1.6e-7, 'float32 rounding floor', fontsize=7.5, color=GREY)
ax.set_xlabel('optimisation step')
# short label: the four-word version collides with the y tick labels at this width
ax.set_ylabel(r'$\|\delta\|/\|\theta\|$')
ax.set_ylim(3e-14, 3.0)
ax.legend(loc='lower right', handlelength=1.2)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

ax = axes[1]
eps = [EPS_VAL[k] for k in EPS_KEYS]
vals = [per_step[k] for k in EPS_KEYS]
ax.semilogx(eps, vals, 'o-', color=C1, ms=3.6)
ax.axhline(1.0, color=GREY, lw=0.7, ls='--')
for x, y in zip(eps, vals):
    ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 6),
                ha='center', fontsize=7.5, color=C1 if y > 1 else GREY)
ax.axvspan(1e-11, 1e-9, color=GREY, alpha=0.07, lw=0)
ax.text(1.4e-10, 0.9955, 'below rounding\nfloor: decays', fontsize=7.5, color=GREY,
        ha='center')
ax.set_xlabel(r'initial perturbation $\varepsilon$ (relative)')
ax.set_ylabel('geometric-mean growth per step')
ax.set_ylim(0.9845, 1.033)
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
fig.tight_layout(w_pad=1.3)
save(fig, 'fig5_amplification')

# ===========================================================================
# Fig 6 - variance removed
# ===========================================================================
cfgs = [('orig\n(grid_sample)', arm(SEED0), C1),
        ('det only\n(+ strict)', arm(DET_ONLY), C3),
        ('sampler\nonly', arm(SAMP_ONLY3), C2),
        ('sampler\n+ strict', arm(SAMP_STRICT), C4)]
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
ax = axes[0]
bins = ['mAP50', '4-8px', '16-32px']
w = 0.2
BASE = 1.0e-3          # bars start here, just below the smallest non-zero sigma;
                       # starting at the axis floor makes them five decades tall and
                       # piles every value label into the same band at the top
for i, (lab, a, c) in enumerate(cfgs):
    vals = [st.stdev(col(a, m)) for m in bins]
    ax.bar(np.arange(3) + (i - 1.5) * w, [max(v, BASE) - BASE for v in vals], w,
           bottom=BASE, color=c, label=lab)
    for xi, v in zip(np.arange(3) + (i - 1.5) * w, vals):
        if v == 0:
            ax.text(xi, BASE * 1.25, '*', ha='center', va='bottom', fontsize=9, color=c)
        else:
            ax.text(xi, v * 1.12, f'{v:.4f}', rotation=90, rotation_mode='anchor',
                    ha='left', va='center', fontsize=7.5, color=c)
ax.set_yscale('log'); ax.set_ylim(BASE * 0.55, 0.35)
ax.set_xticks(np.arange(3)); ax.set_xticklabels([LABEL.get(b, b) for b in bins])
ax.set_ylabel(r'run-to-run $\sigma$ (same-seed repeats)')
# legend above the axes: four entries do not fit inside without covering the bars
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, handlelength=0.9,
          columnspacing=0.6)
ax.text(0.02, 0.03, '$*$ = exactly 0 (bit-identical)', transform=ax.transAxes,
        fontsize=7.5, color=GREY)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

ax = axes[1]
seeds = [3, 4, 5]
o = [arm([f'orig_s{i}'])[f'orig_s{i}']['mAP50'] for i in seeds]
s = [arm([f'detsamp_s{i}'])[f'detsamp_s{i}']['mAP50'] for i in seeds]
for a, b, sd in zip(o, s, seeds):
    ax.plot([0, 1], [a, b], '-', color=GREY, lw=0.7, alpha=0.75)
    ax.plot([0], [a], 'o', color=C1, ms=4)
    ax.plot([1], [b], 's', color=C2, ms=4)
# seeds 3 and 5 land at almost the same mAP50, so their labels are separated
# vertically; a single offset would print them on top of each other.
SEED_OFF = {3: (-6, 7), 4: (-6, 0), 5: (-6, -7)}
for a, sd in zip(o, seeds):
    dx, dy = SEED_OFF[sd]
    ax.annotate(f'seed {sd}', (0, a), textcoords='offset points', xytext=(dx, dy),
                ha='right', va='center', fontsize=7.5, color=GREY)
d = [b - a for a, b in zip(o, s)]
tp = st.mean(d) / (st.stdev(d) / math.sqrt(3))
pp = 2 * stats.t.sf(abs(tp), 2)
ax.set_xticks([0, 1]); ax.set_xticklabels(['F.grid_sample', 'gather sampler'])
ax.set_xlim(-0.42, 1.35)
ax.set_ylim(0.4555, 0.4628)          # headroom so the test result sits clear of the data
ax.set_ylabel(r'mAP$_{50}$ (matched seeds)')
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
ax.text(0.98, 0.97, f'paired mean diff. {st.mean(d):+.4f}\npaired $t={tp:.2f}$, $p={pp:.2f}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=GREY)
fig.tight_layout(w_pad=1.3)
save(fig, 'fig6_variance_removed')

# ===========================================================================
# Fig 7 - early warning
# ===========================================================================
ep = {}
for name in EARLY:
    p = EPOCHS / name / 'results.csv'
    if not p.exists():
        continue
    v = {}
    for line in p.read_text().splitlines()[1:]:
        parts = [q.strip() for q in line.split(',')]
        if len(parts) > 6 and parts[0]:
            try:
                v[int(float(parts[0]))] = float(parts[6])
            except ValueError:
                pass
    ep[name] = v
epochs = sorted({e for v in ep.values() for e in v})
healthy = [n for n in ep if n != BAD]
mu = {e: st.mean([ep[n][e] for n in healthy if e in ep[n]]) for e in epochs}
sg = {e: st.stdev([ep[n][e] for n in healthy if e in ep[n]]) for e in epochs}

fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
ax = axes[0]
for n in healthy:
    ax.plot(list(ep[n]), list(ep[n].values()), color=C1, lw=0.4, alpha=0.28)
ax.plot(list(ep[BAD]), list(ep[BAD].values()), color=C2, lw=1.4, label=BAD)
ax.fill_between(epochs, [mu[e] - 3 * sg[e] for e in epochs], [mu[e] + 3 * sg[e] for e in epochs],
                color=GREY, alpha=0.22, lw=0, label=r'healthy $\mu\pm3\sigma$ (30 runs)')
ax.axvline(2, color=GREY, lw=0.6, ls=':')
# placed in the gap between the bad curve (~0.1 at epoch 12) and the healthy band
# (~0.36), clear of the legend in the lower-right corner
ax.annotate('flagged at\nepoch 2', xy=(2.15, ep[BAD][2]), xytext=(13.0, 0.205),
            fontsize=7.5, color=C2, ha='center',
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color=C2, shrinkA=2, shrinkB=3))
ax.set_xlabel('epoch'); ax.set_ylabel(r'val mAP$_{50}$')
ax.set_ylim(0, 0.43); ax.set_xlim(0.5, 30.5)
# legend above the axes: inside the panel it collides with the annotation
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), handlelength=1.0)
ax.set_title('(a)', loc='left', fontsize=8.2, fontweight='bold')

ax = axes[1]
sub = [e for e in epochs if e <= 5]
for n in healthy:
    y = [ep[n][e] for e in sub if e in ep[n]]
    ax.plot(sub[:len(y)], y, color=C1, lw=0.5, alpha=0.35)
ax.plot(sub, [ep[BAD][e] for e in sub], color=C2, lw=1.4)
ax.fill_between(sub, [mu[e] - 3 * sg[e] for e in sub], [mu[e] + 3 * sg[e] for e in sub],
                color=GREY, alpha=0.22, lw=0)
ax.plot(sub, [mu[e] - 3 * sg[e] for e in sub], color=GREY, lw=0.8, ls='--',
        label=r'$\mu-3\sigma$ threshold')
ax.annotate(f'{ep[BAD][2]:.4f}\n(10.2$\\sigma$ below)', xy=(2.15, ep[BAD][2]), xytext=(3.1, 0.055),
            fontsize=7.5, color=C2,
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color=C2, shrinkA=2, shrinkB=3))
ax.set_xlabel('epoch'); ax.set_ylabel(r'val mAP$_{50}$')
ax.set_ylim(0, 0.43); ax.set_xlim(0.8, 5.2)
ax.legend(loc='upper left', handlelength=1.0)
ax.set_title('(b)', loc='left', fontsize=8.2, fontweight='bold')
fig.tight_layout(w_pad=1.3)
save(fig, 'fig7_early_warning')

# ===========================================================================
# Compliance report: natural widths vs the 390 pt text width, and the implied
# finished lettering size for the smallest text actually set.
# ===========================================================================
print()
print('=' * 78)
print(f'{"figure":<26}{"natural width":>14}{"scale at 390pt":>17}{"8pt prints as":>15}')
print('=' * 78)
import subprocess
for name in ('fig1_overview', 'fig2_runs_per_arm', 'fig3_scale_profile', 'fig4_grad_noise',
             'fig5_amplification', 'fig6_variance_removed', 'fig7_early_warning'):
    p = OUT / f'{name}.pdf'
    out = subprocess.run(['pdfinfo', str(p)], capture_output=True, text=True).stdout
    m = __import__('re').search(r'Page size:\s+([\d.]+) x', out)
    w = float(m.group(1)) if m else float('nan')
    tw = TEXTWIDTH_PT if name != 'fig2_runs_per_arm' else 0.66 * TEXTWIDTH_PT
    sc = tw / w
    print(f'{name:<26}{w:>11.1f} pt{sc:>16.3f}x{8*sc:>13.1f} pt')
print()
print(f'Elsevier rule: finished lettering 7 pt (6 pt floor for sub/superscripts).')
print(f'All figures are authored with no label below 7.5 pt before scaling.')
