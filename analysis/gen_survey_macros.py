#!/usr/bin/env python3
r"""TASK 1: regenerate the \survey* macro block of paper_PR/paper_PR.tex from the
single source of truth, audit/survey_visdrone.csv.

Why this exists
---------------
The manuscript carried two conflicting sets of survey figures: the macro block,
which did not reproduce from the CSV, and Supplementary Table A4, which did.  So
the two could not drift again, the macro block is now GENERATED.  Edit the CSV,
re-run this, and the prose, the abstract, the conclusion and Table A4 all move
together.

The CSV is one row per (paper x metric family), NOT one row per paper
--------------------------------------------------------------------
Seven papers contribute a row to both families (V02/V02b, V04/V04b, V06/V06b,
V08/V08b, V09/V09b, V11/V11b, V12/V12b).  That is why 22 rows is not 22 papers.
Two of the "pairs" are genuinely DIFFERENT works that share a method name --
arXiv:2602.13378 and arXiv:2609.14560 both call themselves LAF-YOLOv10 and
report opposite outcomes on this benchmark -- so the paper key is the resolved
identifier (arxiv_or_doi), never the method name.  Every census count below is
taken after grouping rows to papers.

The derived counts are therefore:
    papers surveyed and verifiable      15
    AP50-family usable deltas            8
    mAP50-95-family usable deltas       10     (9 + QueryDet's second family row)
    restricted AP50 set                  7     (drop the one resolution change)
    state runs / seeds / repeats         3 / 15
    report a standard deviation          3 / 15
    report a confidence interval         0 / 15
    discuss reproducibility              2 / 15

Usage
-----
    python audit/gen_survey_macros.py            # rewrite the block in place
    python audit/gen_survey_macros.py --check    # exit 1 if it is out of date
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'audit' / 'survey_visdrone.csv'
TEX = ROOT / 'paper_PR' / 'paper_PR.tex'

BEGIN = r'%% >>> BEGIN GENERATED SURVEY MACROS'
END = r'%% <<< END GENERATED SURVEY MACROS'

# The one delta produced by changing the input resolution (640 -> 1280) rather
# than by the detector.  Under the manuscript's exclusion rule it leaves the
# restricted AP50 set, which therefore has n = 7.
RESOLUTION_CHANGE_IDS = {'V11'}


def half_up(x: float, places: int = 2) -> str:
    """Round half away from zero, the convention audit/survey_report.md uses.

    The AP50 median is 2.755 exactly; Python's round() is banker's rounding and
    would print 2.75, while the report and every recomputation print 2.76.
    """
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def num(x: str):
    x = (x or '').strip()
    if x in ('', 'n/a', 'NO_DELTA', 'BLOCKED', 'NOT TABLE-VERIFIED', 'EXCLUDED'):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def yes(x: str) -> bool:
    return (x or '').strip().lower() in ('yes', 'true', '1')


def recompute():
    with CSV.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))

    # --- papers: key on the resolved identifier, not the method name ---------
    papers: dict[str, list[dict]] = {}
    for r in rows:
        papers.setdefault(r['arxiv_or_doi'].strip() or r['id'].strip(), []).append(r)

    fam: dict[str, list[float]] = {'AP50-family': [], 'mAP50-95-family': []}
    for r in rows:
        f = r['metric_family'].strip()
        d = num(r['delta_points'])
        if f in fam and d is not None:
            fam[f].append(d)

    def agg(vals):
        v = sorted(vals)
        return dict(n=len(v), median=statistics.median(v), mean=statistics.mean(v),
                    mn=v[0], mx=v[-1], sum=sum(v))

    ap50 = agg(fam['AP50-family'])
    m95 = agg(fam['mAP50-95-family'])

    # restricted AP50: drop the resolution-change entry
    rest_rows = [(r['id'], num(r['delta_points']))
                 for r in rows
                 if r['metric_family'] == 'AP50-family'
                 and r['id'].rstrip('b') not in RESOLUTION_CHANGE_IDS
                 and num(r['delta_points']) is not None]
    rest = agg([v for _, v in rest_rows])

    # --- census, grouped to papers ------------------------------------------
    N = len(papers)
    runs = sum(1 for rs in papers.values() if any(yes(r['runs_stated']) for r in rs))
    std = sum(1 for rs in papers.values() if any(yes(r['std_reported']) for r in rs))
    ci = sum(1 for rs in papers.values() if any(yes(r['ci_reported']) for r in rs))
    repro = sum(1 for rs in papers.values()
                if any(yes(r['reproducibility_discussed']) for r in rs))

    # Denomination notes for the two counts that need one.
    #  * \surveyUnstated counts method papers that state nothing about runs.  The
    #    protocol-defining dataset paper (VisDrone2019-DET, TPAMI 2022) has no
    #    runs of its own to state, so it is outside that denominator:
    #        (N - 1) - runs = 14 - 3 = 11.
    #  * \surveyValSplit counts papers whose record explicitly attributes its
    #    numbers to the val split; the dataset paper is excluded as above, so the
    #    count is method papers on val (9), not all papers on val (10).  This is
    #    the value audit/survey_report.md derives ("9 (of 14 method papers)").
    dataset_papers = [k for k, rs in papers.items()
                      if 'benchmark/protocol-defining' in rs[0]['notes'].lower()
                      or 'not a method paper' in rs[0]['notes'].lower()]
    val_pat = re.compile(r'\bval\b|validation', re.I)
    val_all = [k for k, rs in papers.items()
               if any(val_pat.search(r['split']) for r in rs)]
    val_method = [k for k in val_all if k not in dataset_papers]

    return dict(
        N=N, ap50=ap50, m95=m95, rest=rest,
        runs=runs, std=std, ci=ci, repro=repro,
        unstated=(N - len(dataset_papers)) - runs,
        val_split=len(val_method),
        val_split_all=len(val_all),
        n_dataset=len(dataset_papers),
        nmethod=N - len(dataset_papers),
    )


# \surveyFactor is not a survey count: it is the ratio of the noisiest bin's n=1
# MDE to the survey's median claim, and it is derived from the measurement side
# (10.8 / 2.755 = 3.92 -> 3.9).  It is carried through unchanged.
SURVEY_FACTOR = '3.9'


def block(d, csv_rel: str) -> str:
    a, m, r = d['ap50'], d['m95'], d['rest']
    n = d['N']
    return f'''{BEGIN}
{BEGIN}
%% ---------------------------------------------------------------------
%%  Survey statistics: Section 2.2, the abstract, Section 1, Section 8,
%%  the Conclusion and Supplementary Table A4 (tab:survey).
%%
%%  GENERATED FILE SECTION -- DO NOT EDIT BY HAND.
%%      python audit/gen_survey_macros.py
%%  rewrites everything between the BEGIN/END markers from
%%      {csv_rel}
%%  which is the single source of truth, together with
%%  {csv_rel.replace('survey_visdrone.csv', 'survey_report.md')}.
%%
%%  THE CSV IS ONE ROW PER (PAPER x METRIC FAMILY), NOT ONE ROW PER PAPER.
%%  It has {d['N']} distinct works in 22 rows, because seven papers (V02, V04,
%%  V06, V08, V09, V11, V12) carry a second row for their second metric family.
%%  The paper count and every census count below are therefore obtained by
%%  GROUPING ROWS TO PAPERS FIRST; reading a count off the row list gives the
%%  wrong answer.  Two of the pairs are different works that share a method
%%  name (arXiv:2602.13378 and arXiv:2609.14560 both call themselves
%%  LAF-YOLOv10 and disagree on this benchmark), so a paper is keyed on its
%%  resolved identifier and never on the method name.
%%
%%  Recomputed totals, for reference:
%%    papers surveyed and verifiable .................. {n}
%%    AP50-family usable deltas ....................... {a['n']}
%%    mAP50-95-family usable deltas ................... {m['n']}
%%    restricted AP50 set (one resolution change out) . {r['n']}
%%    state runs / report sd / report CI / discuss repro  {d['runs']} / {d['std']} / {d['ci']} / {d['repro']}  (of {n})
%%
%%  The submitted 24-paper survey was not present in the repository, so the
%%  survey was re-run from primary sources under a stated inclusion rule: a
%%  paper contributes a delta only if BOTH its proposed value and its own
%%  baseline's value were read from a record that was fetched.  The submitted
%%  values (n=24, median +4.40, mean +5.12, range -7.80..+21.70;
%%  mAP50-95 median +2.50, mean +3.52; 3/24 runs, 2/24 sd, 0/24 CI, 12/24 val)
%%  are NOT reproducible from those sources; the values below supersede them.
%%
%%  These macros are the single place the values appear in the prose.
%% ---------------------------------------------------------------------
\\newcommand{{\\surveyN}}{{{n}}}
\\newcommand{{\\surveyDeltaN}}{{{a['n']}}}
\\newcommand{{\\surveyDeltaNA}}{{{m['n']}}}
\\newcommand{{\\surveyMedian}}{{{half_up(a['median'])}}}
\\newcommand{{\\surveyMean}}{{{half_up(a['mean'])}}}
\\newcommand{{\\surveyMin}}{{$-{half_up(abs(a['mn']))}$}}
\\newcommand{{\\surveyMax}}{{$+{half_up(a['mx'])}$}}
\\newcommand{{\\surveyRunsReported}}{{{d['runs']}}}
\\newcommand{{\\surveyUnstated}}{{{d['unstated']}}}
\\newcommand{{\\surveySDReported}}{{{d['std']}}}
\\newcommand{{\\surveyCIReported}}{{{d['ci']}}}
\\newcommand{{\\surveyDiscussRepro}}{{{d['repro']}}}
\\newcommand{{\\surveyRestrictedN}}{{{r['n']}}}
\\newcommand{{\\surveyMedianRestricted}}{{{half_up(r['median'])}}}
\\newcommand{{\\surveyMeanRestricted}}{{{half_up(r['mean'])}}}
\\newcommand{{\\surveyMedianA}}{{{half_up(m['median'])}}}
\\newcommand{{\\surveyMeanA}}{{{half_up(m['mean'])}}}
\\newcommand{{\\surveyValSplit}}{{{d['val_split']}}}
%% ratio of the noisiest bin's n=1 MDE to the survey's median claim
\\newcommand{{\\surveyFactor}}{{{SURVEY_FACTOR}}}
{END}'''


def main() -> int:
    check = '--check' in sys.argv
    d = recompute()
    try:
        rel = CSV.relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(CSV)
    new_block = block(d, rel)

    text = TEX.read_text(encoding='utf-8')
    pat = re.compile(re.escape(BEGIN) + r'.*?' + re.escape(END), re.S)
    m = pat.search(text)
    if not m:
        print(f'FAIL: no generated block markers in {TEX}', file=sys.stderr)
        print(f'      expected {BEGIN} ... {END}', file=sys.stderr)
        return 2

    if check:
        if m.group(0) == new_block:
            print('survey macro block is up to date with the CSV')
            return 0
        print('STALE: the survey macro block does not match the CSV', file=sys.stderr)
        return 1

    TEX.write_text(text[:m.start()] + new_block + text[m.end():], encoding='utf-8')

    a, m95, r = d['ap50'], d['m95'], d['rest']
    print(f'regenerated \\survey* block in {TEX.name} from {rel}')
    print(f'  \\surveyN                {d["N"]}      (papers, not rows)')
    print(f'  \\surveyDeltaN           {a["n"]}      AP50 family')
    print(f'  \\surveyDeltaNA          {m95["n"]}     mAP50-95 family')
    print(f'  \\surveyMedian           {half_up(a["median"])}   = {a["median"]} (mean of the two central values)')
    print(f'  \\surveyMean             {half_up(a["mean"])}   = {a["mean"]} ({a["sum"]:.2f}/{a["n"]})')
    print(f'  \\surveyMin/Max          -{half_up(abs(a["mn"]))} / +{half_up(a["mx"])}')
    print(f'  \\surveyRunsReported     {d["runs"]}')
    print(f'  \\surveyUnstated         {d["unstated"]}     (method papers; the dataset paper has no runs to state)')
    print(f'  \\surveySDReported       {d["std"]}')
    print(f'  \\surveyCIReported       {d["ci"]}')
    print(f'  \\surveyDiscussRepro     {d["repro"]}')
    print(f'  \\surveyRestrictedN      {r["n"]}')
    print(f'  \\surveyMedianRestricted {half_up(r["median"])}   = {r["median"]}')
    print(f'  \\surveyMeanRestricted   {half_up(r["mean"])}   = {r["mean"]:.4f} ({r["sum"]:.2f}/{r["n"]})')
    print(f'  \\surveyMedianA          {half_up(m95["median"])}   = {m95["median"]}')
    print(f'  \\surveyMeanA            {half_up(m95["mean"])}   = {m95["mean"]} ({m95["sum"]:.2f}/{m95["n"]})')
    print(f'  \\surveyValSplit         {d["val_split"]}      (method papers explicitly on val; '
          f'{d["val_split_all"]} including the protocol paper)')
    print(f'  \\surveyFactor           {SURVEY_FACTOR}    (measurement-side ratio, carried through)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
