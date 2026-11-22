"""Applied rate split by whether the target line was among the notes (same tasks as the ans rows)."""
import json, numpy as np
import methods as M
from run import load, recall
from report import latest, vec
logs = load("prefeval", "test")
R = latest("ans", "prefeval", "test")
for name in ["plain", "hyde:mix", "d1:6.ovl", "ceiling"]:
    tops = M.run(name, logs, "prefeval")
    rec = [recall(top, t["target"]) for L, tl in zip(logs, tops) for t, top in zip(L["tasks"], tl)]
    ap = vec(R[name])
    h = [ap[j] for j in ap if rec[j]]; m = [ap[j] for j in ap if not rec[j]]
    print(f"{name:10s} recall {np.mean([rec[j] for j in ap]):.3f}  applied|found {np.mean(h):.3f} (n={len(h)})  applied|missed {np.mean(m) if m else float('nan'):.3f} (n={len(m)})")
