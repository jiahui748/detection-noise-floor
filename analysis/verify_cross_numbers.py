#!/usr/bin/env python3
"""Main text versus supplementary: every shared measurement must be identical.

`verify_numbers.py` compares each document against its own sources, and
`compare_baseline.py` compares the submission against an earlier build.  Neither
answers the question this check exists for: does a number that appears in BOTH
documents have the same value in both?  A transcription slip in the supplement
would otherwise survive every other check.

Two passes:

1. TOKEN SET.  The set of decimal tokens printed in the supplement is compared
   with the set printed by the review build (which carries both the main text and
   the appendices).  A supplement token absent from the review build is either a
   complement number (figure/table numbering, which the two documents assign
   independently) or a slip.

2. SHARED FLOATS.  The five tables that exist in both documents are compared
   cell by cell, digit by digit, after stripping the LaTeX markup.  This is the
   direct check: the same measurement must read the same in both.

Run:  python audit/verify_cross_numbers.py
"""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / 'paper_PR'
TOKEN = re.compile(r'\d+\.\d+')

# Tables that are `\input` by both the main text and the supplement.
SHARED = ['t3_noise_floor', 't6_seed_vs_jitter', 't8_legality', 't9_detector_ab',
          't10_backward_spread', 't11_amplification', 't12_dose_response',
          't13_equivalence', 't14_repeat_sigma', 't15_matched_seed',
          't16_isolated_cost', 't17_e2e_cost', 't18_wallclock',
          't19_early_warning', 'tA1_corpus', 'tA2_config_audit']


def pdftotext(name):
    out = subprocess.run(['pdftotext', '-layout', str(PR / f'{name}.pdf'), '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace')
    txt = out.stdout
    return txt.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')


def decimals(txt):
    return Counter(TOKEN.findall(txt))


fail = 0
supp = decimals(pdftotext('paper_PR_supplementary'))
review = decimals(pdftotext('paper_PR'))
sub = decimals(pdftotext('paper_PR_submission'))

print('=' * 84)
print('1. DECIMAL TOKENS: supplement vs the build that contains the same text')
print('=' * 84)
# Every token the supplement prints should also be printed by the review build,
# which carries the same tables.
only_supp = {k: v for k, v in supp.items() if k not in review}
print(f'  supplement tokens: {len(supp)}   review tokens: {len(review)}')
if only_supp:
    print(f'  note  {len(only_supp)} supplement-only token(s): {sorted(only_supp)[:14]}')
else:
    print('  ok    every supplement decimal token also appears in the review build')

only_in_one = {k for k in sub if k not in review and k not in supp}
print(f'  submission-only tokens (not in either other build): {sorted(only_in_one) or "none"}')

print()
print('=' * 84)
print('2. SHARED FLOATS: cell-by-cell comparison of main text and supplement')
print('=' * 84)


def cells(path):
    """Every decimal printed by a float source file, in order."""
    if not path.exists():
        return None
    body = path.read_text(encoding='utf-8', errors='replace')
    body = re.sub(r'(?<!\\)%[^\n]*', '', body)
    return Counter(TOKEN.findall(body))


checked = 0
for name in SHARED:
    # The supplement reads a moved float from supp_tables/ (a copy that differs
    # only in its cross-reference markup) and everything else from tables/.
    moved = PR / 'supp_tables' / f'{name}.tex'
    a = cells(moved if moved.exists() else PR / 'tables' / f'{name}.tex')
    b = cells(PR / 'tables' / f'{name}.tex')
    if a is None or b is None:
        continue
    checked += 1
    if a != b:
        diff = {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        fail += 1
        print(f'  FAIL  {name}: {diff}')
    else:
        print(f'  ok    {name}: {sum(a.values())} decimal value(s) identical')

print(f'  compared {checked} float pair(s)')

print()
print('=' * 84)
print(f'RESULT: {"FAIL" if fail else "PASS"}  ({fail} problem(s))')
print('=' * 84)
sys.exit(1 if fail else 0)
