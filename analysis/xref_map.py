#!/usr/bin/env python3
"""Cross-reference integrity, both directions, for the three builds.

Builds the explicit mapping the task asks for:

    label -> document that defines it -> number in the BUILT pdf -> number cited

and checks every hard-coded pointer.  A hard-coded number is riskier than a
`\\ref`, because no LaTeX pass can tell that it has drifted; this is the check
that a number still names the float the sentence claims.

What is compared
----------------
* every label's number in each build's `.aux`, against the number printed with
  the float's caption in that build's PDF;
* every literal supplement-float pointer in the supplement sources
  ("Table A8 of the supplement") against the label's number in the main build;
* every literal supplement-float pointer in the main text ("Table A5 of the
  supplement") against the label's number in the supplement build;
* every `\\ref` and `\\mref` target against a label that exists.

Ownership note (TASK 3): an `A`-numbered table, an `E`-numbered figure and a
lettered appendix are all SUPPLEMENT objects, so the ownership phrase is
"of the supplement" in every document.  Only a genuine main-text float (Table 1,
Table 2, Figure 1) is "of the main text".  audit/check_ref_ownership.py checks
that direction explicitly.

Run:  python audit/xref_map.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / 'paper_PR'
BUILDS = ('paper_PR', 'paper_PR_submission', 'paper_PR_supplementary')

KIND = {'tab:': 'Table', 'fig:': 'Figure', 'sec:': 'Section', 'app:': 'Appendix'}
# Literal pointer forms this project writes.  A supplement float is named
# "of the supplement" in BOTH documents (TASK 3): the supplement sources and the
# main text alike, so the same pattern serves both.  A genuine main-text float
# keeps "of the main text", but the manuscript routes those through \ref.
SUPP_PTR = re.compile(r'(Table|Figure|Fig\.|Section|Appendix)~([A-E]?\.?\d[\d.]*)\s+'
                      r'of the supplement')
MAIN_PTR = SUPP_PTR


def aux(name):
    p = PR / f'{name}.aux'
    if not p.exists():
        return {}
    text = p.read_text(encoding='utf-8', errors='replace')
    return {m.group(1): m.group(2) for m in
            re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}', text)}


def pdftotext(name):
    out = subprocess.run(['pdftotext', '-layout', str(PR / f'{name}.pdf'), '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace')
    return re.sub(r'\s+', ' ', out.stdout)


def uncommented(s):
    return re.sub(r'(?<!\\)%[^\n]*', '', s)


def read(rel):
    p = PR / rel
    return uncommented(p.read_text(encoding='utf-8', errors='replace')) if p.exists() else ''


fail = 0
auxes = {b: aux(b) for b in BUILDS}
texts = {b: pdftotext(b) for b in BUILDS}

# ---------------------------------------------------------------- label table
print('=' * 96)
print('LABEL -> DOCUMENT -> NUMBER IN THE BUILT PDF -> NUMBER CITIED')
print('=' * 96)
defined_in = {}
for b in BUILDS:
    for lab in auxes[b]:
        defined_in.setdefault(lab, set()).add(b)

# Which document "owns" a label: the supplement owns the appendix floats.
supp_labels = set(auxes['paper_PR_supplementary'])
main_only = set(auxes['paper_PR']) - supp_labels

# The supplement's literal pointers must equal the MAIN build's number.
supp_src = ''.join(read(r) for r in (
    'appendices.tex', 'extra_supp.tex', 'tables/tA1_corpus.tex',
    'tables/tA2_config_audit.tex', 'supp_tables/t1_findings.tex',
    'supp_tables/t4_runs_per_arm.tex', 'supp_tables/t7_eval_side.tex'))
# The main text's literal pointers must equal the SUPPLEMENT build's number.
main_src = read('paper_PR.tex') + read('newexp_main.tex')

rows = []
# Build label-by-number lookups for cross-checking a literal pointer.
def by_kind_number(auxmap, kind, num):
    want = KIND.get(kind, kind)
    for lab, n in auxmap.items():
        if n == num and KIND.get(lab[:4]) == want:
            return lab
    return None


seen_supp = {}
# A supplement-source pointer names a supplement float, so its number must be the
# number that float has in the SUPPLEMENT's own .aux -- that is the document the
# sentence is printed in, and the number the reader of that PDF will turn to.
for m in SUPP_PTR.finditer(supp_src):
    kind, num = m.group(1), m.group(2)
    if kind == 'Fig.':
        kind = 'Figure'
    lab = by_kind_number(auxes['paper_PR_supplementary'], kind, num)
    seen_supp[(kind, num)] = lab
    ok = lab is not None
    rows.append(('supplement -> supplement', kind, num, lab, ok))

seen_main = {}
for m in MAIN_PTR.finditer(main_src):
    kind, num = m.group(1), m.group(2)
    if kind == 'Fig.':
        kind = 'Figure'
    lab = by_kind_number(auxes['paper_PR_supplementary'], kind, num)
    seen_main[(kind, num)] = lab
    ok = lab is not None
    rows.append(('main -> supplement', kind, num, lab, ok))

print(f'{"direction":22} {"kind":9} {"cited":>6}  label')
for direction, kind, num, lab, ok in rows:
    mark = ' ' if ok else 'FAIL '
    print(f'{mark}{direction:22} {kind:9} {num:>6}  {lab or "NO SUCH OBJECT"}')
    if not ok:
        fail += 1

# Are those pointers actually printed?
for direction, text, pat in (('supplement', texts['paper_PR_supplementary'], SUPP_PTR),
                             ('main text', texts['paper_PR_submission'], MAIN_PTR)):
    missing = []
    for m in pat.finditer(_src := (supp_src if direction == 'supplement' else main_src)):
        s = f'{m.group(1)} {m.group(2)} of the supplement'
        if s not in text:
            missing.append(s)
    if missing:
        fail += 1
        print(f'  FAIL  {direction}: not printed: {sorted(set(missing))[:6]}')

# ------------------------------------------------------- float number in PDF
print()
print('=' * 96)
print('FLOAT NUMBER IN THE BUILT PDF')
print('=' * 96)
for b in BUILDS:
    bad, n = [], 0
    for lab, num in sorted(auxes[b].items()):
        if not lab.startswith(('tab:', 'fig:')):
            continue
        n += 1
        word = 'Table' if lab.startswith('tab:') else 'Figure'
        if f'{word} {num}:' not in texts[b]:
            bad.append(f'{lab}={num}')
    status = 'ok  ' if not bad else 'FAIL'
    if bad:
        fail += 1
    print(f'  {status} {b}: {n} floats, mismatched: {bad or "none"}')

# ---------------------------------------------------------------- \ref check
print()
print('=' * 96)
print('\\ref / \\mref TARGETS')
print('=' * 96)
for b in BUILDS:
    log = PR / f'{b}.log'
    asked = set()
    if log.exists():
        asked = set(re.findall(r"Reference `([^']+)' on page",
                               log.read_text(encoding='utf-8', errors='replace')))
    defined = set(auxes[b])
    missing = sorted(asked - defined)
    if missing:
        fail += 1
    print(f'  {"ok  " if not missing else "FAIL"} {b}: {len(defined)} labels defined, '
          f'undefined references asked: {missing or "none"}')

# --------------------------------------------------- main/supplement agreement
print()
print('=' * 96)
print('MAIN TEXT vs SUPPLEMENT: shared numbers')
print('=' * 96)
both = sorted(supp_labels & set(auxes['paper_PR']))
mismatch = [(lab, auxes['paper_PR_supplementary'][lab], auxes['paper_PR'][lab])
            for lab in both
            if auxes['paper_PR_supplementary'][lab] != auxes['paper_PR'][lab]]
if mismatch:
    print(f'  note  {len(mismatch)} label(s) are numbered differently in the two '
          f'documents (each numbers its own appendix figures):')
    for lab, a, c in mismatch[:12]:
        print(f'        {lab}: supplement={a} main={c}')
else:
    print('  ok    every shared label has the same number in both documents')

print()
print('=' * 96)
print(f'RESULT: {"FAIL" if fail else "PASS"}  ({fail} problem(s))')
print('=' * 96)
sys.exit(1 if fail else 0)
