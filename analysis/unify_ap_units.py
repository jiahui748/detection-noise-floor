#!/usr/bin/env python3
"""Item 6: one canonical form for the AP metric names.

The sources mixed `AP50` / `AP$_{50}$`, `mAP50` / `mAP$_{50}$`, `mAP50-95` /
`mAP$_{50\\text{-}95}$` and a bare "AP points".  The canonical forms are the
subscripted ones, which is the elsarticle norm:

    AP$_{50}$      mAP$_{50}$      mAP$_{50\\text{-}95}$      AP$_{50}$ points

Only the notation changes: no numeric value, no survey macro and no citation is
touched.  Run it AFTER the generators (`make_tables.py`, `make_figures.py`,
`make_exp2b_assets.py`), because it rewrites their output in `paper_PR/tables/`.

Run:  python audit/unify_ap_units.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
# Everything that ships in one of the three builds.
FILES = ['paper_PR.tex', 'newexp_main.tex', 'newexp_section.tex',
         'appendices.tex', 'extra_supp.tex', 'highlights.tex',
         'paper_PR_supplementary.tex', 'paper_PR_submission.tex']
SUBDIRS = ['tables', 'supp_tables']

# Order matters: the two-part metric first, then the single-threshold one.
RULES = [
    (re.compile(r'mAP50-95'), r'mAP$_{50\\text{-}95}$'),
    (re.compile(r'(?<!m)AP50-95'), r'AP$_{50\\text{-}95}$'),
    (re.compile(r'mAP50(?![\d-])'), r'mAP$_{50}$'),
    (re.compile(r'(?<![\w$])AP50(?![\d-])'), r'AP$_{50}$'),
    (re.compile(r'(?<![\w$])AP50-family'), r'AP$_{50}$-family'),
    (re.compile(r'(?<![\w$])AP points\b'), r'AP$_{50}$ points'),
    (re.compile(r'(?<![\w$])mAP points\b'), r'mAP$_{50}$ points'),
]


def targets():
    for rel in FILES:
        p = PR / rel
        if p.exists():
            yield p
    for sub in SUBDIRS:
        for p in sorted((PR / sub).glob('*.tex')):
            yield p


def main():
    before = Counter()
    after = Counter()
    changed = []
    for p in targets():
        text = p.read_text(encoding='utf-8')
        before.update(m.group(0) for r, _ in RULES for m in r.finditer(text))
        new = text
        for pat, rep in RULES:
            new = pat.sub(rep, new)
        if new != text:
            p.write_text(new, encoding='utf-8')
            changed.append(p.relative_to(PR))
        after.update(m.group(0) for r, _ in RULES for m in r.finditer(new))
    print(f'files rewritten: {len(changed)}')
    for c in changed:
        print(f'  {c}')
    print()
    print('bare forms remaining (should be 0):')
    for form in ('mAP50-95', 'AP50-95', 'mAP50', 'AP50', 'AP points', 'mAP points'):
        n = after.get(form, 0)
        flag = '' if n == 0 else '   <-- still present'
        print(f'  {form:12} {n}{flag}')
    print()
    print('canonical forms now:')
    for form in ('mAP$_{50\\text{-}95}$', 'mAP$_{50}$', 'AP$_{50}$'):
        print(f'  {form:26} {after.get(form, 0)}')
    bad = [f for f in ('mAP50-95', 'AP50-95', 'mAP50', 'AP50', 'AP points') if after.get(f)]
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
