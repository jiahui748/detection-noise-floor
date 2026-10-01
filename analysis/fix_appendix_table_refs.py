#!/usr/bin/env python3
"""Route main-text references that live inside the two appendix tables through
\\mainref, so the standalone supplementary build has no dangling references.

tables/tA1_corpus.tex and tA2_config_audit.tex are \\input by appendices.tex and
their captions point at main-text floats (tab:noise_floor, tab:seed_vs_jitter).
Rewriting those to \\mainref is safe for both builds: in the main document
\\mainref is defined as an ordinary \\ref, and in the supplementary driver it
renders as "the main text".

Idempotent.

Run:  python audit/fix_appendix_table_refs.py
"""
import re
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
TABLES = PR / 'tables'
APPENDIX_TABLES = ['tA1_corpus.tex', 'tA2_config_audit.tex']

# labels that the appendix tables may legitimately reference but do not define
MAIN_LABELS = set()
for src in [PR / 'paper_PR.tex'] + sorted(TABLES.glob('*.tex')):
    if src.name in APPENDIX_TABLES:
        continue
    MAIN_LABELS |= set(re.findall(r'\\label\{([^}]+)\}', src.read_text(encoding='utf-8',
                                                                     errors='replace')))

total = 0
for name in APPENDIX_TABLES:
    p = TABLES / name
    txt = p.read_text(encoding='utf-8')
    out, pos, n = [], 0, 0
    for m in re.finditer(r'\\ref\{([^}]+)\}', txt):
        lab = m.group(1)
        out.append(txt[pos:m.start()])
        if lab in MAIN_LABELS:
            out.append(r'\mainref{' + lab + '}')
            n += 1
        else:
            out.append(m.group(0))
        pos = m.end()
    out.append(txt[pos:])
    if n:
        p.write_text(''.join(out), encoding='utf-8')
    total += n
    print(f'{name}: {n} reference(s) routed through \\mainref')

print(f'total: {total}')
