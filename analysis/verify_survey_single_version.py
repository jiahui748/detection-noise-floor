#!/usr/bin/env python3
r"""TASK 5 verification: prove that exactly one survey version is printed.

Every survey figure that reaches a reader is extracted from the built PDFs and
compared with the value recomputed here from audit/survey_visdrone.csv.  This is
the check that the manuscript no longer carries two conflicting sets.

The CSV is one row per (paper x metric family): 22 rows resolve to 15 papers, so
every census count is taken after grouping rows to papers.

Run:  python audit/verify_survey_single_version.py
"""
from __future__ import annotations

import csv
import re
import statistics
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / 'paper_PR'
CSV = ROOT / 'audit' / 'survey_visdrone.csv'
RESOLUTION_CHANGE_IDS = {'V11'}


def half_up(x, places=2):
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places),
                                         rounding=ROUND_HALF_UP))


def num(x):
    x = (x or '').strip()
    if x in ('', 'n/a', 'NO_DELTA', 'BLOCKED', 'NOT TABLE-VERIFIED', 'EXCLUDED'):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def yes(x):
    return (x or '').strip().lower() in ('yes', 'true', '1')


with CSV.open(newline='', encoding='utf-8') as fh:
    ROWS = list(csv.DictReader(fh))

papers = {}
for r in ROWS:
    papers.setdefault(r['arxiv_or_doi'].strip() or r['id'].strip(), []).append(r)


def fam(name):
    return [num(r['delta_points']) for r in ROWS
            if r['metric_family'] == name and num(r['delta_points']) is not None]


AP50, M95 = fam('AP50-family'), fam('mAP50-95-family')
REST = [num(r['delta_points']) for r in ROWS
        if r['metric_family'] == 'AP50-family' and num(r['delta_points']) is not None
        and r['id'].rstrip('b') not in RESOLUTION_CHANGE_IDS]

DATASET = [k for k, rs in papers.items()
           if 'benchmark/protocol-defining' in rs[0]['notes'].lower()]
VAL = re.compile(r'\bval\b|validation', re.I)
VALSET = [k for k, rs in papers.items()
          if any(VAL.search(r['split']) for r in rs) and k not in DATASET]

EXPECT = {
    'papers': len(papers),
    'AP50 n': len(AP50),
    'AP50 median': half_up(statistics.median(AP50)),
    'AP50 mean': half_up(statistics.mean(AP50)),
    'AP50 min': half_up(min(AP50)),
    'AP50 max': half_up(max(AP50)),
    'restricted n': len(REST),
    'restricted median': half_up(statistics.median(REST)),
    'restricted mean': half_up(statistics.mean(REST)),
    'm95 n': len(M95),
    'm95 median': half_up(statistics.median(M95)),
    'm95 mean': half_up(statistics.mean(M95)),
    'runs': sum(1 for rs in papers.values() if any(yes(r['runs_stated']) for r in rs)),
    'sd': sum(1 for rs in papers.values() if any(yes(r['std_reported']) for r in rs)),
    'ci': sum(1 for rs in papers.values() if any(yes(r['ci_reported']) for r in rs)),
    'repro': sum(1 for rs in papers.values()
                 if any(yes(r['reproducibility_discussed']) for r in rs)),
    'val split': len(VALSET),
}

# Macros the generator must have produced.
MACRO_EXPECT = {
    'surveyN': str(EXPECT['papers']),
    'surveyDeltaN': str(EXPECT['AP50 n']),
    'surveyDeltaNA': str(EXPECT['m95 n']),
    'surveyMedian': EXPECT['AP50 median'],
    'surveyMean': EXPECT['AP50 mean'],
    'surveyMin': f"${'-' if float(EXPECT['AP50 min']) < 0 else '+'}"
                 f"{half_up(abs(float(EXPECT['AP50 min'])))}$",
    'surveyMax': f"$+{EXPECT['AP50 max']}$",
    'surveyRestrictedN': str(EXPECT['restricted n']),
    'surveyMedianRestricted': EXPECT['restricted median'],
    'surveyMeanRestricted': EXPECT['restricted mean'],
    'surveyMedianA': EXPECT['m95 median'],
    'surveyMeanA': EXPECT['m95 mean'],
    'surveyRunsReported': str(EXPECT['runs']),
    'surveySDReported': str(EXPECT['sd']),
    'surveyCIReported': str(EXPECT['ci']),
    'surveyDiscussRepro': str(EXPECT['repro']),
    'surveyValSplit': str(EXPECT['val split']),
    'surveyUnstated': str((len(papers) - len(DATASET)) - EXPECT['runs']),
}
SURVEY_FACTOR = '3.9'          # measurement-side ratio, not a survey count

fail = 0
print('=' * 78)
print('VALUES RECOMPUTED FROM', CSV.relative_to(ROOT).as_posix())
print('=' * 78)
print(f'  rows in CSV                              : {len(ROWS)}')
print(f'  distinct papers (rows grouped)           : {EXPECT["papers"]}'
      f'   [22 rows - 7 second-family rows]')
print(f'  AP50 family  n={EXPECT["AP50 n"]:>2}  median {EXPECT["AP50 median"]}  '
      f'mean {EXPECT["AP50 mean"]}  range -{EXPECT["AP50 min"]} .. +{EXPECT["AP50 max"]}')
print(f'  restricted   n={EXPECT["restricted n"]:>2}  median {EXPECT["restricted median"]}  '
      f'mean {EXPECT["restricted mean"]}')
print(f'  mAP50-95     n={EXPECT["m95 n"]:>2}  median {EXPECT["m95 median"]}  '
      f'mean {EXPECT["m95 mean"]}')
print(f'  census  runs {EXPECT["runs"]}  sd {EXPECT["sd"]}  ci {EXPECT["ci"]}  '
      f'repro {EXPECT["repro"]}  val {EXPECT["val split"]}  (of {EXPECT["papers"]})')
print()

# ---------------------------------------------------------------- macro block
print('=' * 78)
print('MACRO BLOCK IN paper_PR.tex')
print('=' * 78)
tex = (PR / 'paper_PR.tex').read_text(encoding='utf-8')
region = re.search(r'BEGIN GENERATED SURVEY MACROS(.*?)END GENERATED SURVEY MACROS',
                   tex, re.S)
if not region:
    print('  FAIL  generated macro block not found')
    fail += 1
    block = ''
else:
    block = region.group(1)
bad = []
for name, want in list(MACRO_EXPECT.items()) + [('surveyFactor', SURVEY_FACTOR)]:
    m = re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', block)
    got = m.group(1) if m else None
    if got != want:
        bad.append((name, want, got))
if bad:
    fail += 1
    print('  FAIL  macro does not match the CSV:')
    for name, want, got in bad:
        print(f'        \\{name}: expected {want!r}, found {got!r}')
else:
    print(f'  ok    all {len(MACRO_EXPECT) + 1} \\survey* macros equal the CSV values')

# ------------------------------------------------------------- printed numbers
print()
print('=' * 78)
print('PRINTED SURVEY NUMBERS, ALL THREE PDFs')
print('=' * 78)


def pdftotext(pdf):
    out = subprocess.run(['pdftotext', '-layout', str(pdf), '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace')
    return re.sub(r'\s+', ' ', out.stdout)


# (regex that must appear, why, builds it applies to)
#   'M' = the main manuscript (review build, submission build)
#   'S' = the standalone supplement
CHECKS = [
    (rf"survey of {EXPECT['papers']} papers", 'abstract: paper count', 'M'),
    (rf"across the {EXPECT['AP50 n']} papers with a computable delta",
     'Section 1: AP50 n', 'M'),
    (rf"median is {EXPECT['AP50 median']} points",
     'Section 1/Section 2: AP50 median', 'M'),
    (rf"median is {EXPECT['m95 median']} points", 'Section 1: m95 median', 'M'),
    (rf"{EXPECT['runs']} of the {EXPECT['papers']} papers state",
     'Section 2: runs count', 'M'),
    (rf"{EXPECT['papers']} papers reporting VisDrone2019-DET accuracy",
     'Section 2: survey size', 'M'),
    (rf"median {EXPECT['restricted median']}", 'Section 2: restricted median', 'M'),
    (rf"mean {EXPECT['restricted mean']}", 'Section 2: restricted mean', 'M'),
    (rf"median {EXPECT['m95 median']} \(mean {EXPECT['m95 mean']}\)",
     'Section 2: m95 values', 'M'),
    (rf"AP\$?_?50 .*median \+{EXPECT['AP50 median']}", 'Table A4: AP50 median', 'S'),
    (rf"median \+{EXPECT['m95 median']}, mean \+{EXPECT['m95 mean']}",
     'Table A4: m95 row', 'S'),
    (rf"{EXPECT['restricted n']}", 'Table A4: restricted n row', 'S'),
    (rf"{EXPECT['runs']}/{EXPECT['papers']}", 'Table A4: runs row', 'S'),
    (rf"mean \+{EXPECT['AP50 mean']}", 'Table A4: AP50 mean', 'S'),
]
# Stale versions that must be gone from every build.  The review build and the
# supplement legitimately quote the SUBMITTED values inside Table A4's caption,
# which exists to say what the recomputed figures supersede; that caption is
# exempt, and its presence is checked separately below.
STALE = [r'14 papers', r'\b5/14\b', r'Table A8 of the main text']
STALE_PAPERS = [r'median \+2\.20', r'mean \+3\.39']
# Table A4's caption is the one place the superseded values are named on purpose.
SUPERSEDED_NOTE = r'These figures replace the submitted values'

for build in ('paper_PR', 'paper_PR_submission', 'paper_PR_supplementary'):
    txt = pdftotext(PR / f'{build}.pdf')
    kind = 'S' if build == 'paper_PR_supplementary' else 'M'
    # The review build carries the manuscript AND the supplement inline, so it
    # must print both sets; the submission drops the appendix; the standalone
    # supplement prints only its own floats.
    want = {'paper_PR': 'MS',
            'paper_PR_submission': 'M',
            'paper_PR_supplementary': 'S'}[build]
    missing = [why for pat, why, for_who in CHECKS
               if for_who in want and not re.search(pat, txt)]
    stale = [p for p in STALE if re.search(p, txt)]
    # The submitted mAP50-95 figures may appear only inside the "we replace the
    # submitted values" caption of Table A4; anywhere else they are a second,
    # conflicting version of the survey.
    for p in STALE_PAPERS:
        for m in re.finditer(p, txt):
            window = txt[max(0, m.start() - 400): m.end() + 400]
            if not re.search(SUPERSEDED_NOTE, window):
                stale.append(f'{p} (outside the superseded-values caption)')
    status = 'ok  ' if not missing and not stale else 'FAIL'
    if missing or stale:
        fail += 1
    print(f'  {status} {build}')
    for w in missing:
        print(f'        MISSING corrected value: {w}')
    for p in stale:
        print(f'        STALE value still printed: {p}')

# \survey macro uses in source, to prove one source feeds all sites.
print()
print('=' * 78)
print('\\survey* MACRO USE SITES')
print('=' * 78)
uses = re.findall(r'\\survey[A-Za-z]+', tex)
sites = {}
for u in uses:
    sites[u] = sites.get(u, 0) + 1
print(f'  {len(uses)} uses of {len(sites)} distinct \\survey* macros in paper_PR.tex')
for name in sorted(sites):
    print(f'    {name:26} {sites[name]}')

print()
print('=' * 78)
print(f'RESULT: {"FAIL" if fail else "PASS"}  ({fail} problem(s))')
print('=' * 78)
sys.exit(1 if fail else 0)
