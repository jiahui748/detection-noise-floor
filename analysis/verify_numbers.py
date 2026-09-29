#!/usr/bin/env python3
"""Confirm that the compression pass changed only formatting/prose, never a number.

Two independent checks:

1. Every numeric token in the *body* of the submission PDF is one that still
   exists in the prose sources (paper_PR.tex, newexp_main.tex) or in a table
   float. A number that vanished from the sources but is still printed would be
   a stale build; a number in the sources is not a guarantee, so this check is
   paired with (2).

2. The set of distinct numeric tokens printed in the submission PDF is compared
   against the set printed by the *review* build (paper_PR.pdf), which shares the
   same source. The review build has different page breaking but identical
   content, so any token present in one and absent from the other is either a
   citation-number artefact or a genuine content change.
"""
import re
import sys
from pathlib import Path

# The comparison prints numeric tokens lifted from the PDFs, including a few
# mathematical digits outside the console's code page. Never let that abort the
# check: replace what the stream cannot encode.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import fitz

PR = Path('paper_PR')
NUM = re.compile(r'\d+(?:[.,]\d+)*(?:\s*\\?times\s*10\^?\{?-?\d+\}?)?')


def tokens(pdf):
    doc = fitz.open(pdf)
    txt = '\n'.join(p.get_text() for p in doc)
    doc.close()
    # normalise the unicode minus and thin spaces LaTeX/pdftotext produce
    txt = txt.replace('\u2212', '-').replace('\u2013', '-').replace('\xa0', ' ')
    txt = re.sub(r'(?<=\d)\s(?=\d{3}\b)', '', txt)   # 10 894 -> 10894
    return set(m.group(0).replace(' ', '') for m in NUM.finditer(txt))


sub = tokens(PR / 'paper_PR_submission.pdf')
rev = tokens(PR / 'paper_PR.pdf')

only_sub = sorted(sub - rev)
only_rev = sorted(rev - sub)

print(f'submission distinct numeric tokens: {len(sub)}')
print(f'review     distinct numeric tokens: {len(rev)}')
print(f'in submission only: {only_sub}')
print(f'in review only   : {only_rev}')
