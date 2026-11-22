"""Retrieval and answer scoring for every memory method on the bed.

  python run.py ret  --slice prefeval --split dev  --methods plain,plain_sent      recall@6 per method
  python run.py ans  --slice prefeval --split dev  --methods none,plain,ceiling --n 60   answer + judge

Splits (PrefEval, 22 logs of 45 tasks): dev = logs 0-3 (180 tasks, knob tuning),
test = logs 4-11 (360 tasks, reported). LoCoMo: all 10 logs (89 tasks), test only.
Every method returns, per task, the 6 lines handed to the model. Rows go to results.jsonl.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import lm
import methods as M
from ret import save

HERE = os.path.dirname(os.path.abspath(__file__))
K = 6


def load(slice_, split):
    L = [json.loads(l) for l in open(os.path.join(HERE, "bed", f"{slice_}.jsonl"))]
    L = [x for x in L if x["tasks"]]
    if slice_ == "prefeval":
        L = L[:4] if split == "dev" else L[4:12] if split == "test" else L[12:]
    return L


def recall(top, tgt):
    return float(any(i in top for i in tgt))


ANS_SYS = ("You are a helpful assistant that remembers earlier conversations with this user. Before each "
           "request you get notes retrieved from your memory of those conversations; some may be irrelevant. "
           "Answer the request in at most 120 words.")
JUDGE_SYS = "You grade assistant replies. Reply with JSON only."
JUDGE = ("For each item below, the user earlier stated a preference, then later made a request. Decide whether "
         "the assistant's reply applies the preference: it recommends nothing that conflicts with the preference "
         "and its suggestions fit the preference (it need not mention it). A reply that ignores the preference, "
         "or includes even one suggestion that conflicts with it, is false. Reply with a JSON list of objects "
         "{{\"id\": <id>, \"reason\": <one short sentence>, \"applied\": true|false}} in the same order.\n\n{items}")
JUDGE_LC = ("For each item below, decide whether the assistant's reply to the question agrees with the reference "
            "answer (same conclusion; wording may differ). Reply with a JSON list of objects "
            "{{\"id\": <id>, \"applied\": true|false}} in the same order.\n\n{items}")


def answer(q, lines):
    notes = "\n".join(f"- {l}" for l in lines) if lines else "(no notes)"
    return lm.call(f"NOTES FROM MEMORY:\n{notes}\n\nUSER REQUEST: {q}", ANS_SYS)


def judge(batch, slice_):
    if slice_ == "prefeval":
        items = "\n\n".join(f"ID {i}\nPREFERENCE: {t['pref']}\nREQUEST: {t['q']}\nREPLY: {a}" for i, (t, a) in batch)
        p = JUDGE.format(items=items)
    else:
        items = "\n\n".join(f"ID {i}\nQUESTION: {t['q']}\nREFERENCE: {t['answer']}\nREPLY: {a}" for i, (t, a) in batch)
        p = JUDGE_LC.format(items=items)
    out = lm.call(p, JUDGE_SYS)
    m = re.search(r"\[.*\]", out, re.S)
    res = {int(r["id"]): bool(r["applied"]) for r in json.loads(m.group(0))}
    return [res.get(i) for i, _ in batch]


def run_ret(a, meths):
    logs = load(a.slice, a.split)
    rows = []
    for name in meths:
        t0 = time.time()
        c0 = dict(lm.STATS)
        tops = M.run(name, logs, a.slice)
        r = [recall(top, t["target"]) for L, tl in zip(logs, tops) for t, top in zip(L["tasks"], tl)]
        dt = time.time() - t0
        row = dict(kind="ret", slice=a.slice + ("" if M.K == 6 else f"@k{M.K}"), split=a.split, method=name, n=len(r), recall6=round(float(np.mean(r)), 4),
                   per_task=r, calls=lm.STATS["calls"] - c0["calls"], sec=round(dt, 1), t=time.strftime("%F %T"))
        rows.append(row)
        print(f"{name:28s} recall@6 {row['recall6']:.3f}  n={row['n']}  new calls {row['calls']}  {dt:.0f}s", flush=True)
    save()
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_ans(a, meths):
    logs = load(a.slice, a.split)
    tasks = [(L, t, i) for L in logs for i, t in enumerate(L["tasks"])]
    # spread the answer sample evenly over logs
    step = max(1, len(tasks) // a.n) if a.n else 1
    pick = set(range(0, len(tasks), step)[: a.n or None])
    for name in meths:
        tops = M.run(name, logs, a.slice)
        flat = [(L, t, top) for L, tl in zip(logs, tops) for t, top in zip(L["tasks"], tl)]
        sel = [(j, x) for j, x in enumerate(flat) if j in pick]

        def go(jx):
            j, (L, t, top) = jx
            lines = [L["lines"][i] for i in top]
            return j, t, answer(t["q"], lines), recall(top, t["target"])

        out = lm.pmap(go, sel, workers=2)
        verd = []
        for b in range(0, len(out), 5):
            chunk = out[b:b + 5]
            verd += judge([(j, (t, ans)) for j, t, ans, _ in chunk], a.slice)
        acc = [float(v) for v in verd if v is not None]
        row = dict(kind="ans", slice=a.slice, split=a.split, method=name, n=len(acc), applied=round(float(np.mean(acc)), 4),
                   per_task={str(j): (None if v is None else float(v)) for (j, _, _, _), v in zip(out, verd)},
                   recall6=round(float(np.mean([r for *_, r in out])), 4), t=time.strftime("%F %T"))
        print(f"{name:28s} applied {row['applied']:.3f}  recall@6 {row['recall6']:.3f}  n={row['n']}", flush=True)
        with open(os.path.join(HERE, "results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ret", "ans"])
    ap.add_argument("--slice", default="prefeval")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--methods", default="plain")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    (run_ret if a.mode == "ret" else run_ans)(a, a.methods.split(","))
    print("lm", lm.STATS)
