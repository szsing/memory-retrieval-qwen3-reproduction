"""Answer check: does better recall turn into answers that respect the hidden fact?

For each request, the design's top-6 lines go to Haiku as "notes about the user" and Haiku answers. A separate judge
call (10 requests per call) sees the hidden fact, the request and the answer, and says whether the answer takes the
fact into account. No design is tuned on this check.

    python3 answer.py plain --beds inmind,implex
    python3 answer.py hyde+prop --mix cpin:1 --w 8 --pv 2 --beds inmind,implex
"""
import argparse
import json
import os
import time

import numpy as np

import bench
import bridge
import rsi
from rsi import K

lm = rsi.lm
HERE = os.path.dirname(os.path.abspath(__file__))

ANS = ("Notes saved about the user from earlier chats (some may be irrelevant):\n{notes}\n\n"
       "The user now asks: {q}\n\nAnswer in at most 80 words.")
JUDGE = ("For each item, a hidden fact about the user is given, then the user's request and an assistant's answer.\n"
         "Say \"yes\" if the answer takes the hidden fact into account (it adapts the advice to it, warns about it, or "
         "avoids what the fact rules out), else \"no\".\n\n{items}\n\n"
         "Reply with a JSON list, one object per item in order: {{\"id\": <n>, \"v\": \"yes\" or \"no\"}}")


def top6(tasks, S):
    out = []
    for t, s in zip(tasks, S):
        p = t["pool"]
        p = np.arange(len(s)) if p is None else (np.arange(*p) if isinstance(p, tuple) else p)
        out.append(p[np.argsort(-s[p])[:K]])
    return out


def beds(names):
    if "inmind" in names:
        lines, nb, tB, _ = bench.bed_inmind_parts()
        tasks = [dict(q=t["q"], targets=t["targets"], pool=None, fact=lines[t["targets"][0]]) for t in tB]
        yield "inmind", lines, tasks
    if "implex" in names:
        b = bench.bed_implex("val")
        users = json.load(open(bench.IMPLEX))["val"]
        facts = [q["trait"] for u in users for q in u["qa"]]
        yield "implex_val", b["lines"], [dict(t, fact=f) for t, f in zip(b["tasks"], facts)]
    if "lucid" in names:
        b = bench.bed_lucid("val")
        yield "lucid_val", b["lines"], [dict(t, fact=" / ".join(b["lines"][i][:300] for i in t["targets"][:2])) for t in b["tasks"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--mix", default="rrf")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--pv", type=int, default=1)
    ap.add_argument("--beds", default="inmind,implex")
    ap.add_argument("--ctrl", default="")
    a = ap.parse_args()
    bench.W["prop"] = a.w; bridge.PV[0] = a.pv; bench.CTRL[0] = a.ctrl
    m = a.mix.split(":"); bench.MIX[0], bench.MIX[1] = m[0], (float(m[1]) if len(m) > 1 else None)
    row = {}
    for name, lines, tasks in beds(set(a.beds.split(","))):
        if a.design == "oracle":          # upper bound: the hidden fact itself is the only note
            notes = [f"- {t['fact']}" for t in tasks]
            hit = [1.0] * len(tasks)
        elif a.design == "none":          # lower bound: no notes
            notes = ["(none)"] * len(tasks)
            hit = [0.0] * len(tasks)
        else:
            S = bench.scores(lines, [t["q"] for t in tasks], set(a.design.split("+")), [t["pool"] for t in tasks])
            tops = top6(tasks, S)
            notes = ["\n".join(f"- {lines[i][:400]}" for i in tp) for tp in tops]
            hit = [float(any(i in tp for i in t["targets"])) for t, tp in zip(tasks, tops)]
        answers = lm.pmap(lambda x: lm.call(ANS.format(notes=x[0], q=x[1])), list(zip(notes, [t["q"] for t in tasks])))

        def judge(ch):
            items = "\n\n".join(f"Item {i + 1}\nHidden fact: {t['fact'][:400]}\nRequest: {t['q']}\nAnswer: {ans}" for i, (t, ans) in enumerate(ch))
            return [str(v).strip().lower().startswith("y") for v in bridge.by_id(bridge.jcall(JUDGE.format(items=items)), len(ch), "v")]
        ok = bridge.batched(list(zip(tasks, answers)), 10, judge)
        row[name] = dict(recall=int(sum(hit)), applied=int(sum(ok)), n=len(tasks), applied_given_hit=int(sum(o and h for o, h in zip(ok, hit))))
        print(f"[{a.design} {a.mix} {a.ctrl} w={a.w}] {name}: recall {row[name]['recall']}/{len(tasks)}  answer uses fact {row[name]['applied']}/{len(tasks)}  (of hits: {row[name]['applied_given_hit']})", flush=True)
    with open(os.path.join(HERE, "answer_runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M"), design=a.design, mix=a.mix, ctrl=a.ctrl, w=a.w, pv=a.pv, res=row)) + "\n")


if __name__ == "__main__":
    main()
