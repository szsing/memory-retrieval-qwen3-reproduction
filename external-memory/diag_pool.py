"""Pool diagnostics for D1: share of log lines tagged, pool size per request, target-in-pool rate."""
import json, sys, numpy as np
import methods as M
from run import load
for sl, sp in [("prefeval", "test"), ("locomo", "test"), ("len960", "test")]:
    tagged, pool, inpool = [], [], []
    for L in load(sl, sp):
        lc = M.LG(M.gen_cats_lines, L, L["lines"]); qc = M.QG(M.gen_cats_reqs, L, [t["q"] for t in L["tasks"]])
        lcs = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in lc]
        tagged.append(np.mean([bool(c) for c in lcs]))
        for t, c in zip(L["tasks"], qc):
            c = set(int(x) for x in (c or []) if str(x).isdigit())
            p = [i for i, l in enumerate(lcs) if c & l]
            pool.append(len(p) / len(lcs)); inpool.append(any(i in p for i in t["target"]))
    print(f"{sl:9s} lines tagged {np.mean(tagged):.2f}  pool share median {np.median(pool):.3f}  target in pool {np.mean(inpool):.2f}")
