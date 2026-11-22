"""Scale grid on log length: derived logs built from the 8 test logs (4-11), no new model calls.

  len120  first 120 lines of each test log, tasks whose buried line falls inside   (8 logs)
  len480  two test logs back to back                                               (4 logs, 90 tasks each)
  len960  four test logs back to back                                              (2 logs, 180 tasks each)
Each line and request keeps a pointer to its source log ("src", "tsrc"), so the model notes written for the
240-line logs are reused as they are (methods.LG / QG).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = [json.loads(l) for l in open(os.path.join(HERE, "bed", "prefeval.jsonl"))][4:12]


def cut(L, n):
    tasks, tsrc = [], []
    for j, t in enumerate(L["tasks"]):
        if all(i < n for i in t["target"]):
            tasks.append(t)
            tsrc.append((L["log"], j))
    return dict(log=L["log"] + f"c{n}", lines=L["lines"][:n], src=[(L["log"], j) for j in range(n)], tasks=tasks,
                tsrc=tsrc)


def cat(Ls):
    lines, src, tasks, tsrc = [], [], [], []
    for L in Ls:
        off = len(lines)
        lines += L["lines"]
        src += [(L["log"], j) for j in range(len(L["lines"]))]
        for j, t in enumerate(L["tasks"]):
            tasks.append(dict(t, target=[i + off for i in t["target"]]))
            tsrc.append((L["log"], j))
    return dict(log="+".join(L["log"] for L in Ls), lines=lines, src=src, tasks=tasks, tsrc=tsrc)


beds = {"len120": [cut(L, 120) for L in T],
        "len480": [cat(T[i:i + 2]) for i in range(0, 8, 2)],
        "len960": [cat(T[i:i + 4]) for i in range(0, 8, 4)]}
for k, Ls in beds.items():
    with open(os.path.join(HERE, "bed", k + ".jsonl"), "w") as f:
        for L in Ls:
            f.write(json.dumps(L) + "\n")
    print(k, len(Ls), "logs", sum(len(L["tasks"]) for L in Ls), "tasks", len(Ls[0]["lines"]), "lines each")
