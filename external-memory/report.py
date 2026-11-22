"""Tables from results.jsonl: recall@6 (or applied rate) per method, and paired 95% bootstrap intervals.

  python report.py --slice prefeval --split test --vs hyde:mix
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def latest(kind, slice_, split):
    R = {}
    for line in open(os.path.join(HERE, "results.jsonl")):
        r = json.loads(line)
        if r["kind"] == kind and r["slice"] == slice_ and r["split"] == split:
            R[r["method"]] = r
    return R


def vec(r):
    p = r["per_task"]
    if isinstance(p, dict):
        return {int(k): v for k, v in p.items() if v is not None}
    return dict(enumerate(p))


def paired(a, b, B=4000):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks], float)
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="ret")
    ap.add_argument("--slice", default="prefeval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--vs", default="hyde:mix")
    a = ap.parse_args()
    R = latest(a.kind, a.slice, a.split)
    key = "recall6" if a.kind == "ret" else "applied"
    base = vec(R[a.vs]) if a.vs in R else None
    print(f"{'method':24s} {key:>8s}   n   minus {a.vs} [95% CI]")
    for m, r in sorted(R.items(), key=lambda x: x[1][key]):
        line = f"{m:24s} {r[key]:8.3f} {r['n']:4d}"
        if base is not None and m != a.vs:
            d, lo, hi, n = paired(vec(r), base)
            line += f"   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(line)
