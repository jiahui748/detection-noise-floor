#!/usr/bin/env python3
"""Recompute every VisDrone-survey number directly from audit/survey_visdrone.csv.

This is the single source of truth. Nothing here reads a macro, a table or a
prose value from the manuscript.

Key structural fact: the CSV is ONE ROW PER (paper x metric family), not one row
per paper.  Seven papers contribute a row to BOTH families, so 22 rows resolve
to 15 distinct papers.
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

CSV = Path(__file__).resolve().parent / "survey_visdrone.csv"

# The one row whose delta is produced by changing the input resolution
# (640 -> 1280) rather than by the detector; excluded from the restricted set.
RESOLUTION_CHANGE_IDS = {"V11"}


def num(x: str):
    x = (x or "").strip()
    if x in ("", "n/a", "NO_DELTA", "BLOCKED", "NOT TABLE-VERIFIED", "EXCLUDED"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def yes(x: str) -> bool:
    return (x or "").strip().lower() in ("yes", "true", "1")


def main() -> int:
    with CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print("=" * 78)
    print("RAW CSV SHAPE")
    print("=" * 78)
    print(f"rows                              : {len(rows)}")

    # --- paper identity -----------------------------------------------------
    # A "paper" is a distinct work. Rows sharing a leading id stem (V02/V02b)
    # are the same paper read for a second family. V08/V09 are DIFFERENT works
    # that share a method name; they carry different ids AND different arXiv
    # identifiers, so key on the resolved identifier, falling back to the id.
    def paper_key(r):
        return r["arxiv_or_doi"].strip() or r["id"].strip()

    papers = {}
    for r in rows:
        papers.setdefault(paper_key(r), []).append(r)
    print(f"distinct works (papers)           : {len(papers)}")
    print(f"distinct CSV ids (row stems)      : {len({r['id'].rstrip('b') for r in rows})}")
    print()

    fam = {"AP50-family": [], "mAP50-95-family": []}
    for r in rows:
        f = r["metric_family"].strip()
        d = num(r["delta_points"])
        if f in fam and d is not None:
            fam[f].append((r["id"], r["short_name"], d))

    def stats(name, items):
        vals = sorted(v for _, _, v in items)
        n = len(vals)
        med = statistics.median(vals)
        mean = statistics.mean(vals)
        print("-" * 78)
        print(f"{name}: n={n}")
        print("-" * 78)
        for rid, sn, v in sorted(items, key=lambda t: t[2]):
            print(f"  {rid:<5} {v:+7.2f}   {sn}")
        print(f"  sorted : {', '.join(f'{v:+.2f}' for v in vals)}")
        print(f"  sum    : {sum(vals):+.4f}")
        print(f"  median : {med:+.4f}  -> {med:+.2f}")
        print(f"  mean   : {mean:+.4f}  -> {mean:+.2f}")
        print(f"  min    : {vals[0]:+.2f}   max: {vals[-1]:+.2f}")
        print()
        return dict(n=n, vals=vals, median=med, mean=mean, mn=vals[0], mx=vals[-1])

    ap50 = stats("AP50 family", fam["AP50-family"])
    m95 = stats("mAP50-95 family", fam["mAP50-95-family"])

    # --- restricted AP50 set ------------------------------------------------
    rest = [(i, s, v) for (i, s, v) in fam["AP50-family"]
            if i.rstrip("b") not in RESOLUTION_CHANGE_IDS]
    print("=" * 78)
    print("RESTRICTED AP50 SET  (drop resolution-change entries)")
    print("=" * 78)
    print(f"  exclusion rule: drop CSV id(s) {sorted(RESOLUTION_CHANGE_IDS)} "
          f"(input 640 -> 1280)")
    rstats = stats("Restricted AP50 family", rest)

    # same rule applied to the mAP50-95 family (for symmetry)
    m95rest = [(i, s, v) for (i, s, v) in fam["mAP50-95-family"]
               if i.rstrip("b") not in RESOLUTION_CHANGE_IDS]
    mstats_r = stats("Restricted mAP50-95 family (symmetry)", m95rest)

    # --- census: grouped to papers, never per row ---------------------------
    print("=" * 78)
    print("CENSUS (per PAPER, rows grouped first)")
    print("=" * 78)
    census = {}
    for key, rs in papers.items():
        census[key] = dict(
            label=rs[0]["short_name"],
            ids=[r["id"] for r in rs],
            split=" | ".join(sorted({r["split"] for r in rs})),
            runs=any(yes(r["runs_stated"]) for r in rs),
            std=any(yes(r["std_reported"]) for r in rs),
            ci=any(yes(r["ci_reported"]) for r in rs),
            repro=any(yes(r["reproducibility_discussed"]) for r in rs),
        )
    N = len(census)
    runs = sum(1 for c in census.values() if c["runs"])
    std = sum(1 for c in census.values() if c["std"])
    ci = sum(1 for c in census.values() if c["ci"])
    repro = sum(1 for c in census.values() if c["repro"])
    print(f"  papers surveyed and verifiable          : {N}")
    print(f"  state runs / seeds / repeats            : {runs}  "
          f"({', '.join(c['label'] for c in census.values() if c['runs'])})")
    print(f"  report a standard deviation or +/-      : {std}  "
          f"({', '.join(c['label'] for c in census.values() if c['std'])})")
    print(f"  report a confidence interval            : {ci}")
    print(f"  discuss reproducibility / seed variance : {repro}  "
          f"({', '.join(c['label'] for c in census.values() if c['repro'])})")
    print(f"  state nothing about runs                : {N - runs}")

    def val_like(s: str) -> bool:
        s = s.lower()
        return bool(re.search(r"\bval\b|validation", s))
    valp = [c for c in census.values() if val_like(c["split"])]
    print(f"  evaluate on the val split (explicit)    : {len(valp)}")
    print(f"  split not stated at all                 : "
          f"{sum(1 for c in census.values() if 'not stated' in c['split'].lower())}")
    print()

    # --- family assignment sanity ------------------------------------------
    print("=" * 78)
    print("FAMILY ASSIGNMENT AUDIT")
    print("=" * 78)
    print(f"  papers with an AP50 row      : {len({r['id'].rstrip('b') for r in rows if r['metric_family']=='AP50-family'})}")
    print(f"  papers with a mAP50-95 row   : {len({r['id'].rstrip('b') for r in rows if r['metric_family']=='mAP50-95-family'})}")
    print(f"  papers with NEITHER          : "
          f"{sorted({r['id'] for r in rows if r['metric_family'] not in ('AP50-family','mAP50-95-family')})}")
    both = {r["id"].rstrip("b") for r in rows if r["metric_family"] == "AP50-family"} & \
           {r["id"].rstrip("b") for r in rows if r["metric_family"] == "mAP50-95-family"}
    print(f"  papers with BOTH families    : {sorted(both)}  (count {len(both)})")
    print(f"  rows accounted for           : "
          f"{len({r['id'] for r in rows})} ids + {len(both)} secondary rows "
          f"= {len({r['id'] for r in rows}) + len(both)}")
    print()

    # --- factor -------------------------------------------------------------
    print("=" * 78)
    print("DERIVED")
    print("=" * 78)
    print(f"  surveyFactor (MDE / median)  : see §8; not derived here")
    print()
    print("AUTHORITATIVE VALUES")
    print(f"  surveyN                 = {N}")
    print(f"  surveyDeltaN  (AP50)    = {ap50['n']}")
    print(f"  surveyDeltaNA (mAP50-95)= {m95['n']}")
    print(f"  surveyMedian            = {ap50['median']:.2f}")
    print(f"  surveyMean              = {ap50['mean']:.2f}")
    print(f"  surveyMin               = {ap50['mn']:.2f}")
    print(f"  surveyMax               = {ap50['mx']:.2f}")
    print(f"  surveyRunsReported      = {runs}")
    print(f"  surveyUnstated          = {N - runs}")
    print(f"  surveySDReported        = {std}")
    print(f"  surveyCIReported        = {ci}")
    print(f"  surveyDiscussRepro      = {repro}")
    print(f"  surveyRestrictedN       = {rstats['n']}")
    print(f"  surveyMedianRestricted  = {rstats['median']:.2f}")
    print(f"  surveyMeanRestricted    = {rstats['mean']:.2f}")
    print(f"  surveyMedianA           = {m95['median']:.2f}")
    print(f"  surveyMeanA             = {m95['mean']:.2f}")
    print(f"  surveyValSplit          = {len(valp)}")
    print(f"  (restricted mAP50-95: n={mstats_r['n']} median={mstats_r['median']:.2f} "
          f"mean={mstats_r['mean']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
