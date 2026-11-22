"""Build the bed: chat logs of about 240 user lines, each with indirect requests whose answer depends on
one earlier line.

  python bed.py            writes bed/prefeval.jsonl and bed/locomo.jsonl

PrefEval slice (implicit, persona-driven form): each of the 1,000 PrefEval items is a short persona chat
in which the user mentions a preference in passing (for example a gluten-free diet while asking about
night-shift snacks), plus a later request that never names it (cafes to visit in San Francisco).
One log = 45 such chats in a row (about 240 user lines), so every log holds 45 facts and 45 requests
(many facts per user). The target line is the user turn that states the preference: the turn
closest to the preference sentence under BGE-small (the request is never used to pick it).

LoCoMo slice: the 96 category-3 questions (inference, e.g. "Would Caroline likely have Dr. Seuss books
on her bookshelf?"), each conversation cut to 240 lines that keep every evidence turn.
Order is fixed by hashing, so the bed is the same on every machine.
"""
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "bed")
PER_LOG = 45


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def prefeval(emb):
    items = []
    for f in sorted(glob.glob(os.path.join(D, "persona", "*.json"))):
        topic = os.path.basename(f)[:-5]
        for x in json.load(open(f)):
            c = x["conversation"]
            us = [c[k]["user"].replace("\n", " ").strip() for k in sorted(c, key=int)]
            items.append(dict(topic=topic, pref=x["preference"], q=x["question"], why=x.get("explanation", ""), turns=us))
    items.sort(key=lambda x: h(x["pref"] + x["q"]))
    # target turn = closest to the preference sentence
    allturns = [t for x in items for t in x["turns"]]
    E = emb(allturns)
    P = emb([x["pref"] for x in items])
    i = 0
    for k, x in enumerate(items):
        e = E[i:i + len(x["turns"])]
        i += len(x["turns"])
        s = e @ P[k]
        x["tgt_local"] = int(s.argmax())
        x["tgt_cos"] = float(s.max())
    logs = []
    for L in range(len(items) // PER_LOG):
        chunk = items[L * PER_LOG:(L + 1) * PER_LOG]
        lines, tasks = [], []
        for x in chunk:
            base = len(lines)
            lines += x["turns"]
            tasks.append(dict(q=x["q"], target=[base + x["tgt_local"]], pref=x["pref"], why=x["why"],
                              topic=x["topic"], tgt_cos=round(x["tgt_cos"], 3)))
        logs.append(dict(log=f"pe{L:02d}", lines=lines, tasks=tasks))
    return logs


def locomo():
    d = json.load(open(os.path.join(D, "locomo10.json")))
    logs = []
    for x in d:
        conv = x["conversation"]
        turns = []
        s = 1
        while f"session_{s}" in conv:
            date = conv.get(f"session_{s}_date_time", "")
            for t in conv[f"session_{s}"]:
                turns.append((t["dia_id"], f"[{date}] {t['speaker']}: {t['text']}".replace("\n", " ")))
            s += 1
        qs = [q for q in x["qa"] if q["category"] == 3]
        ev = {e for q in qs for e in q["evidence"]}
        keep = [t for t in turns if t[0] in ev]
        rest = sorted([t for t in turns if t[0] not in ev], key=lambda t: h(t[1]))
        keep_ids = {t[0] for t in keep} | {t[0] for t in rest[:max(0, 240 - len(keep))]}
        lines = [t for t in turns if t[0] in keep_ids]
        idx = {t[0]: i for i, t in enumerate(lines)}
        tasks = [dict(q=q["question"], target=[idx[e] for e in q["evidence"] if e in idx], answer=str(q["answer"]))
                 for q in qs]
        tasks = [t for t in tasks if t["target"]]
        logs.append(dict(log=x["sample_id"], lines=[t[1] for t in lines], tasks=tasks))
    return logs


if __name__ == "__main__":
    from ret import embed
    os.makedirs(OUT, exist_ok=True)
    pe = prefeval(embed)
    with open(os.path.join(OUT, "prefeval.jsonl"), "w") as f:
        for L in pe:
            f.write(json.dumps(L) + "\n")
    lc = locomo()
    with open(os.path.join(OUT, "locomo.jsonl"), "w") as f:
        for L in lc:
            f.write(json.dumps(L) + "\n")
    cos = [t["tgt_cos"] for L in pe for t in L["tasks"]]
    print(f"prefeval: {len(pe)} logs, {sum(len(L['tasks']) for L in pe)} tasks, "
          f"{np.mean([len(L['lines']) for L in pe]):.0f} lines/log; target cos median {np.median(cos):.2f}, "
          f"<0.6: {sum(c < 0.6 for c in cos)}")
    print(f"locomo: {len(lc)} logs, {sum(len(L['tasks']) for L in lc)} tasks, "
          f"{np.mean([len(L['lines']) for L in lc]):.0f} lines/log")
