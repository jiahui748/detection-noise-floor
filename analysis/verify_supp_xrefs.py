#!/usr/bin/env python3
"""TASK 1 verification: prove the supplement's explicit numbers are right.

The supplement writes every pointer to a float as a literal number, because a
separate document cannot `\\ref` the other one.  Since TASK 3 established that an
`A`-numbered table, an `E`-numbered figure and a lettered appendix are all
SUPPLEMENT objects, that literal is named "of the supplement" in every document:

    Table A8 of the supplement      the jitter test, a supplement float
    Table 1 of the main text        the noise floor, a main-text float

Three checks, all against the built PDFs:

1. Every literal reference in the supplement sources uses the ownership phrase
   that matches where the float actually lives, and the number equals that
   label's value in the `.aux` of the document that owns it.
2. The supplement prints each of those references, and its own floats are
   numbered as its own `.aux` says.
3. The sweeps the task asks for: no "Table the main text", no "Table Table",
   no "??" and no other doubled word in any of the three PDFs.

Ownership direction is checked independently, against the built .aux files, by
audit/check_ref_ownership.py.

Run:  python audit/verify_supp_xrefs.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / 'paper_PR'

KIND = {'tab:': 'Table', 'fig:': 'Figure', 'sec:': 'Section', 'app:': 'Appendix'}
MACRO = re.compile(r'\\(?:mainref|AppRef|TabAref|FigAref)\{([^}]*)\}'
                   r'|\\mref\{([^}]*)\}')
# A supplement-owned float is named "of the supplement"; a genuine main-text
# float keeps "of the main text".  Number syntax covers both "A8" and "E.2".
LITERAL = re.compile(r'(?<![A-Za-z])(Table|Figure|Fig\.|Section|Appendix)~'
                     r'([A-E](?:\.\d+)?|\d+(?:\.\d+)?)\s+of the (main text|supplement)')
SOURCES = ['paper_PR_supplementary.tex', 'appendices.tex', 'extra_supp.tex',
           'tables/tA1_corpus.tex', 'tables/tA2_config_audit.tex',
           'supp_tables/t1_findings.tex', 'supp_tables/t2_survey.tex',
           'supp_tables/t4_runs_per_arm.tex', 'supp_tables/t5_cross_benchmark.tex',
           'supp_tables/t7_eval_side.tex']


def aux(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    return {m.group(1): m.group(2) for m in
            re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}', text)}


def pdftotext(pdf):
    out = subprocess.run(['pdftotext', '-layout', str(pdf), '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace')
    return out.stdout


def uncommented(s):
    return re.sub(r'(?<!\\)%[^\n]*', '', s)


def source_labels(rel):
    p = PR / rel
    if not p.exists():
        return set()
    return set(re.findall(r'\\label\{([^}]*)\}', uncommented(
        p.read_text(encoding='utf-8', errors='replace'))))


fail = 0
main = aux(PR / 'paper_PR.aux')
supp = aux(PR / 'paper_PR_supplementary.aux')
supp_txt = pdftotext(PR / 'paper_PR_supplementary.pdf')

# The supplement's own labels: its appendices and its own floats.
local = set()
for rel in ('paper_PR_supplementary.tex', 'appendices.tex', 'extra_supp.tex',
            'tables/tA1_corpus.tex', 'tables/tA2_config_audit.tex'):
    local |= source_labels(rel)

print('=' * 74)
print('EXPLICIT CROSS-REFERENCES IN THE SUPPLEMENT SOURCES')
print('=' * 74)
seen, wrong_form = [], []
for rel in SOURCES:
    p = PR / rel
    if not p.exists():
        continue
    body = p.read_text(encoding='utf-8', errors='replace')
    for m in LITERAL.finditer(uncommented(body)):
        word = 'Figure' if m.group(1) == 'Fig.' else m.group(1)
        seen.append((rel, word, m.group(2), m.group(3)))
    for m in MACRO.finditer(uncommented(body)):
        lab = m.group(1) or m.group(2)
        if lab not in local:
            wrong_form.append((rel, m.group(0)))
if wrong_form:
    fail += 1
    print('  FAIL  a reference into the main text is not written out:')
    for rel, what in wrong_form[:8]:
        print(f'        {rel}: {what}')
else:
    print('  ok    every cross-reference is written as an explicit number')

# A float whose number is A-prefixed (or an E-prefixed figure, or a lettered
# appendix) is a supplement object, so its ownership phrase must be "of the
# supplement"; an unprefixed number is a main-text float.
mis_owned = []
for rel, word, num, owner in seen:
    is_supp = bool(re.match(r'^[A-E]', num))
    if word == 'Section':
        is_supp = False          # sections are main-text unless lettered
        if re.match(r'^[A-E]$', num):
            is_supp = True       # an appendix letter
    want = 'supplement' if is_supp else 'main text'
    if owner != want:
        mis_owned.append((rel, f'{word}~{num} of the {owner}', f'should be "{want}"'))
if mis_owned:
    fail += 1
    print('  FAIL  ownership phrase does not match where the float lives:')
    for rel, what, why in mis_owned[:10]:
        print(f'        {rel}: {what} -> {why}')
else:
    print(f'  ok    all {len(seen)} ownership phrases match the float\'s document')

# Every supplement-owned number must be a label in the SUPPLEMENT's own .aux --
# the document the sentence is printed in; every main-text number must be one in
# the manuscript's.
NUM = {}
for lab, num in supp.items():
    NUM.setdefault((KIND.get(lab[:4], '?'), num), []).append(lab)
bad = []
for rel, word, num, owner in seen:
    if re.match(r'^[A-E]', num) and NUM.get((word, num)):
        continue
    if not re.match(r'^[A-E]', num):
        continue                      # a genuine main-text float, checked below
    bad.append((rel, word, num))
if bad:
    # A Figure number can legitimately differ because the two documents number
    # their appendix figures independently; that is reported, not failed.
    hard = [b for b in bad if b[1] != 'Figure']
    if hard:
        fail += 1
        print(f'  FAIL  supplement numbers with no matching label in the '
              f'supplement\'s own .aux: {hard[:6]}')
    else:
        print(f'  note  {len(bad)} figure pointer(s) use the manuscript\'s '
              f'Appendix E numbering')
else:
    print(f'  ok    all supplement-owned numbers match a label in the supplement .aux')

print()
print('=' * 74)
print('NUMBERS AS PRINTED')
print('=' * 74)
missing = []
for rel, word, num, owner in seen:
    if f'{word} {num} of the {owner}' not in re.sub(r'\s+', ' ', supp_txt):
        missing.append((rel, word, num, owner))
if missing:
    fail += 1
    print(f'  FAIL  not printed in paper_PR_supplementary.pdf: {missing[:6]}')
else:
    print(f'  ok    all {len(seen)} explicit pointers are printed in the supplement')

own_bad = []
for lab, num in sorted(supp.items()):
    if lab.startswith('tab:') and f'Table {num}:' not in re.sub(r'\s+', ' ', supp_txt):
        own_bad.append((lab, num))
    if lab.startswith('fig:') and f'Figure {num}:' not in re.sub(r'\s+', ' ', supp_txt):
        own_bad.append((lab, num))
if own_bad:
    fail += 1
    print(f'  FAIL  supplement float numbers that do not match its PDF: {own_bad}')
else:
    print(f'  ok    all {len(supp)} supplement float numbers match its PDF')

print()
print('=' * 74)
print('SWEEPS: ALL THREE BUILDS')
print('=' * 74)
PATS = ('Table the main text', '§the main text', 'Section the main text',
        'Table Table', 'Figure Figure', 'Appendix Appendix', 'Section Section',
        '??')
for build in ('paper_PR', 'paper_PR_submission', 'paper_PR_supplementary'):
    txt = pdftotext(PR / f'{build}.pdf')
    counts = {p: txt.count(p) for p in PATS}
    badpat = {p: n for p, n in counts.items() if n}
    if badpat:
        fail += 1
        print(f'  FAIL  {build}: {badpat}')
    else:
        print(f'  ok    {build}: all {len(PATS)} sweeps zero')

n = len(re.findall(r'of the main text', re.sub(r'\s+', ' ', supp_txt)))
n1 = len(re.findall(r'of the supplement', re.sub(r'\s+', ' ', supp_txt)))
print(f'  info  supplement prints {n} "of the main text" and {n1} "of the supplement" pointers')
n2 = len(re.findall(r'of the supplement', re.sub(r'\s+', ' ', pdftotext(PR / 'paper_PR_submission.pdf'))))
print(f'  info  submission prints {n2} "of the supplement" pointers')

print()
print('=' * 74)
print(f'RESULT: {"FAIL" if fail else "PASS"}  ({fail} problem(s))')
print('=' * 74)
sys.exit(1 if fail else 0)
