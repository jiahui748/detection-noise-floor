"""Compare the decimal-token counters of the pre-cut baseline against the current build.

BASELINE_paper_PR_submission.txt is a Counter dump of every decimal token printed in
the submission build before this pass. If this pass removed only prose, then every key
of the baseline must still be present with a count no smaller than the baseline's.
"""
import ast
import re
import sys
from collections import Counter

import pymupdf

TOKEN = re.compile(r'\d+\.\d+')


def decimals(pdf):
    doc = pymupdf.open(pdf)
    txt = '\n'.join(p.get_text() for p in doc)
    doc.close()
    txt = txt.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
    return Counter(TOKEN.findall(txt))


base = Counter(dict(ast.literal_eval(
    open('paper_PR/BASELINE_paper_PR_submission.txt', encoding='utf-8').read())))
cur = decimals('paper_PR/paper_PR_submission.pdf')

lost = {k: (v, cur.get(k, 0)) for k, v in base.items() if cur.get(k, 0) < v}
gained = {k: v for k, v in cur.items() if k not in base}

print(f'decimal tokens: baseline {len(base)}, current {len(cur)}')
print(f'baseline total occurrences: {sum(base.values())}, current: {sum(cur.values())}')
print(f'\nTOKENS WHOSE COUNT FELL (possible lost numbers): {len(lost)}')
for k, (b, c) in sorted(lost.items()):
    print(f'  {k:>12}  baseline x{b} -> now x{c}')
print(f'\nTOKENS NEW SINCE THE BASELINE (expected: the Section 5.4 addition): {len(gained)}')
for k, v in sorted(gained.items()):
    print(f'  {k:>12}  x{v}')
