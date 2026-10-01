#!/usr/bin/env python3
"""Aggregate the exp1 result JSONs: same-seed sigma per arm, and bit-identity check.

The primary outcome is sigma across same-seed repeats -- NOT accuracy.  This script
therefore reports sigma for every metric and additionally reports whether the repeat
weights are bit-identical (equal sha256), which is the stronger statement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = ["mAP50", "mAP50_95", "mAP75"]


def stdev(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def clean(v):
    """COCO reports -1 for an undefined metric (no ground truth in that area bin).
    Those sentinels must not enter a sigma, or a bin with no objects would look like
    a perfect measurement."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="<REMOTE-ROOT>/pr_exp/exp1/result")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    res = Path(args.result)
    runs = []
    for p in sorted(res.glob("exp1_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}")
            continue
        if d.get("status") != "ok":
            print(f"skip {p.name}: status={d.get('status')}")
            continue
        d["_file"] = p.name
        runs.append(d)

    if not runs:
        print("no completed runs yet")
        return

    arms = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r)

    print(f"{'arm':>4} {'n':>3} {'sampler':>11} {'strict':>6} "
          f"{'mAP50 mean':>11} {'sd':>9} {'mAP50-95 mean':>14} {'sd':>9} "
          f"{'same hash':>10}")
    summary = {}
    for arm in sorted(arms):
        rs = arms[arm]
        m50 = [v for v in (clean(r.get("mAP50")) for r in rs) if v is not None]
        m5095 = [v for v in (clean(r.get("mAP50_95")) for r in rs) if v is not None]
        hashes = {r.get("weights_sha256") for r in rs}
        mean50 = sum(m50) / len(m50) if m50 else float("nan")
        mean5095 = sum(m5095) / len(m5095) if m5095 else float("nan")
        sd50, sd5095 = stdev(m50), stdev(m5095)
        print(f"{arm:>4} {len(rs):>3} {rs[0]['sampler']:>11} {str(rs[0]['strict']):>6} "
              f"{mean50:>11.5f} {(sd50 if sd50 is not None else float('nan')):>9.5f} "
              f"{mean5095:>14.5f} {(sd5095 if sd5095 is not None else float('nan')):>9.5f} "
              f"{str(len(hashes) == 1):>10}")
        summary[arm] = {
            "n": len(rs),
            "sampler": rs[0]["sampler"],
            "strict": rs[0]["strict"],
            "mAP50_mean": mean50, "mAP50_sd": sd50, "mAP50_values": m50,
            "mAP50_95_mean": mean5095, "mAP50_95_sd": sd5095, "mAP50_95_values": m5095,
            "distinct_weight_hashes": len(hashes),
            "bit_identical": len(hashes) == 1 and len(rs) > 1,
            "hashes": sorted(h for h in hashes if h),
            "epochs": rs[0]["epochs"],
            "seconds_per_epoch": rs[0].get("seconds_per_epoch"),
            "grid_sample_calls_probe_forward": rs[0].get("grid_sample_calls_probe_forward"),
        }
        # per-size sigma, which is what the paper's small-object claim needs
        sizes = {}
        keys = set()
        for r in rs:
            keys |= set((r.get("by_size") or {}).keys())
        for k in sorted(keys):
            vals50 = [v for v in (clean(r["by_size"][k]["mAP50"]) for r in rs
                                  if r.get("by_size") and k in r["by_size"]) if v is not None]
            vals95 = [v for v in (clean(r["by_size"][k]["mAP50_95"]) for r in rs
                                  if r.get("by_size") and k in r["by_size"]) if v is not None]
            sizes[k] = {"mAP50_mean": sum(vals50) / len(vals50) if vals50 else None,
                        "mAP50_sd": stdev(vals50),
                        "mAP50_95_mean": sum(vals95) / len(vals95) if vals95 else None,
                        "mAP50_95_sd": stdev(vals95),
                        "n_defined": len(vals95)}
        summary[arm]["by_size"] = sizes

    print()
    print("PER-SIZE sigma (mAP50-95):")
    for arm in sorted(summary):
        b = summary[arm]["by_size"]
        parts = [f"{k}={b[k]['mAP50_95_sd']:.5f}" for k in sorted(b)
                 if b[k]["mAP50_95_sd"] is not None]
        print(f"  arm {arm}: " + "  ".join(parts))

    print()
    print("VERDICT")
    a, d = summary.get("a"), summary.get("d")
    if a and a["n"] >= 2:
        print(f"  arm a (grid_sample, strict off): sigma(mAP50) = {a['mAP50_sd']:.5f}"
              f"  over n={a['n']}  distinct hashes={a['distinct_weight_hashes']}")
    elif a:
        print(f"  arm a: only n={a['n']} run(s) so far -- sigma needs at least 2. "
              f"mAP50 = {a['mAP50_mean']:.5f}")
    if d and d["n"] >= 2:
        print(f"  arm d (gather, strict on)      : sigma(mAP50) = {d['mAP50_sd']:.5f}"
              f"  over n={d['n']}  distinct hashes={d['distinct_weight_hashes']}"
              f"  bit_identical={d['bit_identical']}")
    c = summary.get("c")
    if c and c["n"] >= 2:
        print(f"  arm c (gather, strict off)     : sigma(mAP50) = {c['mAP50_sd']:.5f}"
              f"  over n={c['n']}  distinct hashes={c['distinct_weight_hashes']}")

    out = Path(args.json_out) if args.json_out else res.parent / "exp1_sigma.json"
    out.write_text(json.dumps({"arms": summary, "n_runs": len(runs)}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
