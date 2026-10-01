#!/usr/bin/env python3
"""Finish the appendix split: route every appendix reference to a *main-text*
label through \\mainref, and define that macro for the standalone supplementary.

`split_submission.py` classified labels by scanning paper_PR.tex alone, which
misses every label that lives inside an \\input-ed float file (tables/*.tex,
figures). This pass scans those too, so the supplementary file has no dangling
references.

Idempotent: safe to re-run.

Run:  python audit/fix_mainrefs.py
"""
import re
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
APP = PR / 'appendices.tex'
MAIN = PR / 'paper_PR.tex'

# labels that exist in the main document: the main .tex plus every float it inputs
main_labels = set()
for src in [MAIN] + sorted((PR / 'tables').glob('*.tex')):
    main_labels |= set(re.findall(r'\\label\{([^}]+)\}', src.read_text(encoding='utf-8',
                                                                     errors='replace')))
app = APP.read_text(encoding='utf-8')
app_labels = set(re.findall(r'\\label\{([^}]+)\}', app))
main_labels -= app_labels               # defined locally in the appendix: keep as \ref

n = 0
out = []
pos = 0
for m in re.finditer(r'\\ref\{([^}]+)\}', app):
    lab = m.group(1)
    out.append(app[pos:m.start()])
    if lab in main_labels:
        out.append(r'\mainref{' + lab + '}')
        n += 1
    else:
        out.append(m.group(0))
    pos = m.end()
out.append(app[pos:])
APP.write_text(''.join(out), encoding='utf-8')

# the main document needs \mainref defined; the supplementary defines its own
if r'\providecommand{\mainref}' not in MAIN.read_text(encoding='utf-8'):
    raise SystemExit('paper_PR.tex is missing the \\providecommand{\\mainref} definition')

leftover = re.findall(r'\\ref\{([^}]+)\}', APP.read_text(encoding='utf-8'))
print(f'rewrote {n} appendix reference(s) to main-text labels as \\mainref')
print(f'remaining \\ref in appendices.tex: {len(leftover)} '
      f'(all appendix-local: {sorted(set(leftover))})')
