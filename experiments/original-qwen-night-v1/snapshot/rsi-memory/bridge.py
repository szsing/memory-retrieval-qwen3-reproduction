"""New designs after the 21-design search (2026-11-16 night). Static designs scored once on InMind.

  python bridge.py <design> [--w W]

Failure found in the 21-design search: the 22 unseen-domain tasks every design missed need a hidden middle fact
about the THING in the request (capsules contain gelatin; hot dogs are a choking risk; antihistamines sedate),
which then clashes with the user's fact (halal; a two-year-old; a pilot medical exam). Topic labels cannot join
them: the request is "shopping/medicine", the fact is "religion".

Design "prop" (property join): both sides are rewritten into the same middle space of concrete properties.
  write time: each stored fact -> properties of things it makes unsafe, forbidden, risky or special
  read time:  each request     -> hidden properties of the things it involves
  score:      dense + w * max cosine(request property, line property)
Scored on: testB (61 unseen-domain tasks, official 241-line log), testA, naive (harm), as in rsi.py.
"""
import argparse
import os
import sys

import numpy as np

import rsi
from rsi import jcall, by_id, batched, numbered, embed, Store, mean, log, lm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external-memory"))

PROPL = ("Below are lines a user wrote to a chat assistant. For each line that states a lasting fact about the user "
         "(a condition, diet, belief, job, family member, age, legal or money situation), list up to 8 short, concrete "
         "properties of products, foods, ingredients, substances, activities, places, rules or offers that this fact "
         "makes unsafe, forbidden, risky, restricted or specially treated for this user. Write properties, not "
         "requests. Example: \"I wear a pacemaker\" -> [\"strong magnets\", \"MRI scan\", \"induction cooktop\", "
         "\"electromagnetic interference\", \"airport body scanner\", \"electric muscle stimulation\"]. For lines "
         "with no lasting fact give [].\nReply with a JSON list of objects {{\"id\": <id>, \"p\": [<properties>]}}."
         "\n\nLINES:\n{lines}")

PROPQ = ("Below are requests a user made to a chat assistant. For each request, list up to 8 short, concrete hidden "
         "properties of the things it involves that could make the right answer different for some people: "
         "ingredients, materials, hazards, side effects, legal, tax or eligibility rules, and kinds of people it is "
         "unsafe or restricted for. Example: \"Recommend a scented candle for my bedroom\" -> [\"open flame\", "
         "\"fire hazard\", \"paraffin smoke\", \"fragrance allergens\", \"toxic to cats\", \"asthma trigger\"].\n"
         "Reply with a JSON list of objects {{\"id\": <id>, \"p\": [<properties>]}}.\n\nREQUESTS:\n{reqs}")


PROPL2 = ("Below are lines a user wrote to a chat assistant. For each line that states something lasting about the user "
          "(a condition, diet, belief, job, family member, age, place, money or legal situation, or a strong like, "
          "dislike or habit, even in passing), list up to 8 short, concrete properties (1-4 words each) of products, "
          "foods, ingredients, substances, activities, places, rules or offers that this makes unsafe, forbidden, "
          "unwanted, preferred or specially treated for this user. Write properties, not requests. Examples: \"I wear "
          "a pacemaker\" -> [\"strong magnets\", \"MRI scan\", \"induction cooktop\", \"airport body scanner\"]; "
          "\"... we avoid crowded places like beachfront hotels\" -> [\"crowds\", \"noisy venues\", \"beachfront "
          "hotels\", \"tourist hotspots\"]. For lines with nothing lasting about the user give [].\nReply with a JSON "
          "list of objects {{\"id\": <id>, \"p\": [<properties>]}}.\n\nLINES:\n{lines}")

PROPQ2 = ("Below are requests a user made to a chat assistant. For each request, list up to 8 short, concrete hidden "
          "properties (1-4 words each) of the things it involves that could make the right answer different for some "
          "people: ingredients, materials, hazards, side effects, noise, crowds, cost, legal, tax or eligibility "
          "rules, and kinds of people it is unsafe, restricted or unpleasant for. Example: \"Recommend a scented "
          "candle for my bedroom\" -> [\"open flame\", \"fire hazard\", \"paraffin smoke\", \"fragrance "
          "allergens\", \"toxic to cats\", \"strong scent\"].\nReply with a JSON list of objects {{\"id\": <id>, "
          "\"p\": [<properties>]}}.\n\nREQUESTS:\n{reqs}")
PV = [1]   # prompt version: 1 = round-2 prompts, 2 = also likes and dislikes, 1-4 word properties


def props_lines(lines):
    def go(ch):
        j = jcall((PROPL2 if PV[0] == 2 else PROPL).format(lines=numbered([l[:600] for l in ch])))
        return [[x for x in (v or []) if isinstance(x, str)][:8] for v in by_id(j, len(ch), "p")]
    return batched(list(lines), 30, go)


PROPQ_GATE = PROPQ.replace("Reply with a JSON list", "Give [] for a request that does not ask the assistant for "
                           "advice, a recommendation, a plan or a task done for the user (for example a question "
                           "about past events, about what someone said, or about other people).\n"
                           "Reply with a JSON list")
QGATE = [False]


def props_reqs(qs):
    def go(ch):
        j = jcall((PROPQ2 if PV[0] == 2 else PROPQ_GATE if QGATE[0] else PROPQ).format(reqs=numbered(ch)))
        return [[x for x in (v or []) if isinstance(x, str)][:8] for v in by_id(j, len(ch), "p")]
    return batched(list(qs), 8, go)


GATEQ = ("Below are messages a user sent to a chat assistant. For each message, decide whether the user asks the "
         "assistant to help with the user's own life now: advice, a recommendation, a choice, a plan, or a task done for "
         "the user (\"help\"), or asks to recall or look up information, such as what happened, what someone said or "
         "did, or facts about other people (\"recall\").\nReply with a JSON list of objects {{\"id\": <id>, \"k\": "
         "\"help\" or \"recall\"}}.\n\nMESSAGES:\n{reqs}")


def gate_reqs(qs):
    """True = the request asks for help with the user's own life (the property join applies)."""
    def go(ch):
        j = jcall(GATEQ.format(reqs=numbered(ch)))
        return [str(v).strip().lower() != "recall" for v in by_id(j, len(ch), "k")]
    return batched(list(qs), 20, go)


MIX = "sum"
CTRL = ""
LLMSEL = False
SIDE = "both"


class PropIndex:
    def __init__(self, line_props):
        flat, own = [], []
        for i, ps in enumerate(line_props):
            for p in ps:
                flat.append(p)
                own.append(i)
        self.E = embed(flat)
        self.own = np.array(own, dtype=int)
        self.n = len(line_props)

    def bonus(self, req_props):
        out = []
        for ps in req_props:
            b = np.zeros(self.n)
            if ps:
                sim = (self.E @ embed(ps).T).max(1)
                np.maximum.at(b, self.own, sim)
            out.append(b)
        return out


def score_set(st, tasks, pidx, w, use_d1, naive=False):
    qs = [t["naive"] if naive else t["q"] for t in tasks]
    qt1 = None
    if use_d1:
        import methods as Mx
        qt1 = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in Mx.gen_cats_reqs(qs)]
    S = st.scores(qs, None, None, qt1)
    if LLMSEL:   # rival: the model reads the whole store (all lines) and names the lines each request needs
        picks = rsi.miss_finder(st, qs)
        S = [s + 10.0 * np.isin(np.arange(len(s)), [p for p in pk if p < len(s)]) for s, pk in zip(S, picks)]
    if pidx is not None and w:
        rp = props_reqs(qs) if SIDE != "line" else [[q] for q in qs]   # line-only: the raw request meets line properties
        if CTRL == "shuf":      # control: each request gets another request's properties
            h_ = len(rp) // 2; rp = rp[h_:] + rp[:h_]
        B = pidx.bonus(rp)
        if CTRL == "null":      # control: constant bonus for every line that has any property (a "fact line" prior)
            has = np.zeros(pidx.n); has[pidx.own] = 1.0
            B = [has.copy() for _ in rp]
        if MIX == "max":
            S = [np.maximum(s, w * b) for s, b in zip(S, B)]
        elif MIX == "rrf":
            def rr(x):
                r = np.empty(len(x)); r[np.argsort(-x)] = np.arange(len(x)); return 1.0 / (60 + r)
            S = [rr(s) + w * rr(b) for s, b in zip(S, B)]
        else:
            S = [s + w * b for s, b in zip(S, B)]
    return st.recall(tasks, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--bed", default="inmind", choices=["inmind", "prefeval"])
    ap.add_argument("--side", default="both", choices=["both", "line", "req"])
    ap.add_argument("--ctrl", default="", choices=["", "null", "shuf"])
    ap.add_argument("--mix", default="sum", choices=["sum", "max", "rrf"])
    ap.add_argument("--w", type=float, default=None, help="fixed weight; default: pick on A (stream+gate)")
    a = ap.parse_args()
    global MIX, CTRL, SIDE, LLMSEL
    MIX, CTRL, SIDE = a.mix, a.ctrl, a.side
    parts = set(a.design.split("+"))
    LLMSEL = "llmsel" in parts
    bed = rsi.bed_inmind() if a.bed == "inmind" else rsi.bed_prefeval()
    st = Store(bed)
    c0 = lm.STATS["calls"] + lm.STATS["cached"]
    use_d1 = "d1" in parts
    if use_d1:
        import methods as Mx
        lc = Mx.gen_cats_lines([l[:600] for l in st.lines])
        st.d1l = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in lc]
    if "prop" in parts:
        lp = props_lines(st.lines) if SIDE != "req" else [[l[:600]] for l in st.lines]  # req-only: request properties meet raw lines
        pidx = PropIndex(lp)
    else:
        pidx = None
    w = a.w
    if pidx is not None and w is None:   # weight chosen on learning-domain tasks only
        dev = bed["stream"] + bed["gate"]
        best = None
        for cand in (0.25, 0.5, 1.0, 2.0):
            d, o = score_set(st, dev, pidx, cand, use_d1)
            sc = mean(d) + (mean(o) or 0)
            print(f"  dev w={cand}: dense {mean(d)} off {mean(o)}")
            if best is None or sc > best[0]:
                best = (sc, cand)
        w = best[1]
    res = {}
    for name in ("testA", "testB"):
        if not bed[name]:
            res[name] = (None, None, [])
            continue
        d, o = score_set(st, bed[name], pidx, w, use_d1)
        res[name] = (mean(d), mean(o), o)
    dn, on = score_set(st, bed["testA"] + bed["testB"], pidx, w, use_d1, naive=True)
    calls = lm.STATS["calls"] + lm.STATS["cached"] - c0
    hits = int(sum(res["testB"][2])) if res["testB"][2] else int(round(res["testB"][0] * len(bed["testB"])))
    print(f"[{a.design}:{MIX}{CTRL}:{SIDE}] w={w} A off {res['testA'][1]} B off {res['testB'][1]} ({hits}/{len(bed['testB'])}) B dense {res['testB'][0]} "
          f"naive {mean(on)} calls {calls}")
    log(dict(kind="round", bed=a.bed, design=a.design + ("" if MIX == "sum" else ":" + MIX) + (":" + CTRL if CTRL else "") + ("" if SIDE == "both" else ":" + SIDE + "only"), round=0, w=w, testA_off=res["testA"][1],
             testA_dense=res["testA"][0], testB_off=res["testB"][1], testB_dense=res["testB"][0],
             testB_per=res["testB"][2], naive_off=mean(on), calls_total=calls, rules=[], ntags=0, note="bridge.py"))


if __name__ == "__main__":
    main()


PICK = """A user asks an assistant: {q}

Notes saved from the user's earlier chats (most are unrelated):
{notes}

Which notes could change what a good answer should say (a fact, situation, limit, habit, like or dislike of this
user that matters here)? Reply with JSON only: {{"pick": [note numbers, most important first, at most 6]}}"""


def pick_notes(qs, cand_texts):
    """For each request, Haiku picks up to 6 candidate notes that matter (returns 0-based indices into its candidates)."""
    def one(x):
        q, cs = x
        j = jcall(PICK.format(q=q, notes="\n".join(f"{i + 1}. {c[:300]}" for i, c in enumerate(cs))))
        v = j.get("pick", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
        out = []
        for n in v:
            try:
                n = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= n < len(cs) and n not in out:
                out.append(n)
        return out[:6]
    return lm.pmap(one, list(zip(qs, cand_texts)))
