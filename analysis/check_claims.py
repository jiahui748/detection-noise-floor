#!/usr/bin/env python3
"""Check the rewritten manuscript against the audit: retracted claims must be gone,
corrected numbers must be present, and every float must be referenced.

Run:  python audit/check_claims.py
"""
import re
import sys
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
TEX = PR / 'paper_PR.tex'
TABLES = PR / 'tables'

# (regex, why it must NOT appear)  -- claims the audit retracted
FORBIDDEN = [
    (r'2\.9\s*(?:to|-|\u2013)\s*10\.2', 'retracted noise-floor ratio range (is 2.2 to 10.2)'),
    (r'load-bearing measurement', 'the "load-bearing measurement" framing of the retracted F test'),
    (r'F\s*=\s*0\.42,\s*p\s*=\s*0\.011', 'the retracted approximation p-value'),
    (r'p\s*<\s*0\.001\s*;\s*F\s*=\s*0\.42', 'the retracted approximation p-value'),
    (r'one run out of seven', 'retracted bad-run denominator (is eight)'),
    (r'1\s+in\s+7\s+sampler', 'retracted bad-run denominator'),
    (r'2\.9 to 10\.2', 'retracted ratio range'),
    (r'differ by \$?1\.2\s*\\times\s*10\^\{-9\}\$? relative\b(?!.*range)',
     'the single-value framing of the operator noise'),
    (r'Fig\.?\s*~?\\?ref\{fig:amplification\}[^.]{0,40}relative parameter distance',
     'sanity placeholder (should not fire)'),
]

# (regex, why it MUST appear)
REQUIRED = [
    (r'0\.00299|0\.003', 'corrected sampler-only sigma'),
    (r'1\.31|1\.309|t\s*=\s*1\.3', 'paired t statistic'),
    (r'p\s*=\s*0\.32', 'paired p-value'),
    (r'9\.9\s*\\times\s*10\^\{-9\}|9\.9\\times ?10\^\{-9\}', 'upper end of the operator noise range'),
    (r'2\.2', 'corrected ratio range lower bound'),
    (r'off by a factor of two|factor of two|too small by', 'documented p-value formula error'),
    (r'seed contributes no detectable variance', 'the surviving central claim'),
    (r'paired', 'paired test used'),
    (r'8 runs|eight runs|1/8|one out of eight|one in eight', 'corrected bad-run denominator'),
]

fail = 0
if not TEX.exists():
    raise SystemExit(f'missing {TEX}')
text = TEX.read_text(encoding='utf-8', errors='replace')

print('=' * 74)
print('FORBIDDEN (retracted) claims')
print('=' * 74)
for pat, why in FORBIDDEN:
    hits = [m.group(0) for m in re.finditer(pat, text, re.I | re.S)]
    if hits:
        fail += 1
        print(f'  FAIL  {why}\n        found: {hits[:3]}')
    else:
        print(f'  ok    absent: {why}')

print()
print('=' * 74)
print('REQUIRED (corrected) content')
print('=' * 74)
for pat, why in REQUIRED:
    hits = re.findall(pat, text, re.I)
    if hits:
        print(f'  ok    present: {why}  ({len(hits)}x)')
    else:
        fail += 1
        print(f'  FAIL  MISSING: {why}')

print()
print('=' * 74)
print('CONTEXT-SENSITIVE: the retracted 0.00025 value')
print('=' * 74)
# The retracted sigma may appear ONLY as part of an explanation of the error, never as
# a live value. Flag any occurrence not within 200 characters of a corrective word.
EXPLAIN = re.compile(r'wrong|instead|listed|error|retract|two replicates|generator|glob',
                     re.I)
bad = []
for m in re.finditer(r'0\.00025', text):
    window = text[max(0, m.start() - 200): m.end() + 200]
    if not EXPLAIN.search(window):
        bad.append(text[max(0, m.start() - 90): m.end() + 90].replace('\n', ' '))
if bad:
    fail += 1
    print(f'  FAIL  {len(bad)} occurrence(s) presented as a live value:')
    for b in bad:
        print(f'        ...{b}...')
else:
    print('  ok    every occurrence is inside an explanation of the error')

print()
print('=' * 74)
print('FLOAT WIRING')
print('=' * 74)


def uncommented(s):
    """Drop LaTeX comment bodies: a `\\ref{#1}` inside a \\providecommand is not a
    reference, and the file's own header documents the macros in comments."""
    return re.sub(r'(?<!\\)%[^\n]*', '', s)


inputs = set(re.findall(r'\\input\{tables/([^}]+)\}', text))
labels_defined = set()
for f in TABLES.glob('*.tex'):
    labels_defined |= set(re.findall(r'\\label\{([^}]+)\}',
                                     uncommented(f.read_text(encoding='utf-8', errors='replace'))))
labels_defined |= set(re.findall(r'\\label\{([^}]+)\}', uncommented(text)))
# The supplement sources name main-text labels through \mref; those labels live
# in the main text (and its \input-ed experiment section), not in the appendix.
for extra in ('appendices.tex', 'extra_supp.tex', 'newexp_main.tex', 'newexp_section.tex'):
    p = TABLES.parent / extra
    if p.exists():
        labels_defined |= set(re.findall(r'\\label\{([^}]+)\}',
                                         uncommented(p.read_text(encoding='utf-8',
                                                                errors='replace'))))
# \mref carries a main-text label into the supplement; in this build it is a \ref,
# so its targets must resolve here too.  `#1` is a macro parameter, not a label.
refs = set(r for r in re.findall(r'\\ref\{([^}]+)\}', uncommented(text)) if r != '#1')
for extra in ('appendices.tex', 'extra_supp.tex', 'newexp_main.tex', 'newexp_section.tex'):
    p = TABLES.parent / extra
    if p.exists():
        body = uncommented(p.read_text(encoding='utf-8', errors='replace'))
        refs |= set(r for r in re.findall(r'\\ref\{([^}]+)\}', body) if r != '#1')
        refs |= set(re.findall(r'\\mref\{([^}]+)\}', body))
unref = sorted(l for l in labels_defined if l.startswith('tab:') and l not in refs)
undefined = sorted(r for r in refs if r not in labels_defined)
print(f'  tables generated : {len(list(TABLES.glob("*.tex")))}')
print(f'  tables \\input    : {len(inputs)}')
print(f'  \\ref/\\mref targets: {len(refs)}')
if unref:
    print(f'  NOTE  tables never \\ref\'d in prose: {unref}')
if undefined:
    fail += 1
    print(f'  FAIL  \\ref to undefined label: {undefined}')
else:
    print('  ok    every \\ref resolves')

missing_tables = sorted(f'{n}.tex' for n in inputs if not (TABLES / f'{n}.tex').exists())
if missing_tables:
    fail += 1
    print(f'  FAIL  \\input of missing file: {missing_tables}')

print()
print('=' * 74)
print(f'RESULT: {"FAIL" if fail else "PASS"}  ({fail} problem(s))')
print('=' * 74)
sys.exit(1 if fail else 0)
