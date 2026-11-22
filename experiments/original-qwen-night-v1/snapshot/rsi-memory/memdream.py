"""MemDream-style baseline on InMind (prompt-only; MemDream 2609.34545 trains its agents with GRPO, we do not).

  python memdream.py <design> [--rounds 2] [--bed inmind|prefeval]     design parts: d1, prop (joined by +)

Dreamer: for each stored line with a lasting user fact, write 3 probe requests, including indirect ones.
Analyst: retrieve each probe. Source line outside the top 6 -> "buried" (recall failure). A line that sits in the
         top 6 of 3+ probes from other lines -> "noisy" (precision failure).
Consolidator: buried line -> extract up to 4 memory notes that make it findable (stored as extra keys of the line);
              noisy line -> pruned by a score penalty (size picked on learning-domain tasks, never on test).
Repeat the dream on still-buried lines for --rounds rounds. Scoring and logging as in bridge.py; with "prop" the
property join is added on top (rank fusion, w picked on learning-domain tasks).
"""
import argparse
from collections import Counter

import numpy as np

import bridge
import rsi
from rsi import jcall, by_id, batched, numbered, Store, mean, log, lm, K

DREAM = ("Below are lines a user wrote to a chat assistant, stored in its memory. For each line that states a lasting "
         "fact about the user, write 3 requests the user might make much later that this memory should help answer. "
         "Make at least 2 of them indirect: everyday requests that do not mention the fact but whose best answer "
         "depends on it. For lines with no lasting fact give [].\n"
         "Reply with a JSON list of objects {{\"id\": <id>, \"q\": [<requests>]}}.\n\nLINES:\n{lines}")

CONS = ("Below are memory entries of a chat assistant, each with requests it should have been found for but was not "
        "(retrieval is by embedding similarity). For each entry, write up to 4 short memory notes that restate what "
        "the entry implies so that a search for requests like these finds it. Notes must stay true to the entry.\n"
        "Reply with a JSON list of objects {{\"id\": <id>, \"notes\": [<notes>]}}.\n\nENTRIES:\n{items}")


def dream(lines):
    def go(ch):
        j = jcall(DREAM.format(lines=numbered([l[:600] for l in ch])))
        return [[x for x in (v or []) if isinstance(x, str)][:3] for v in by_id(j, len(ch), "q")]
    return batched(list(lines), 30, go)


def consolidate(items):
    def go(ch):
        txt = "\n".join(f"{i}. ENTRY: {l[:400]}\n   MISSED FOR: " + " | ".join(ps) for i, (l, ps) in enumerate(ch))
        j = jcall(CONS.format(items=txt))
        return [[x for x in (v or []) if isinstance(x, str)][:4] for v in by_id(j, len(ch), "notes")]
    return batched(list(items), 10, go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--bed", default="inmind", choices=["inmind", "prefeval"])
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    parts = set(a.design.split("+"))
    bridge.MIX = "rrf"
    bed = rsi.bed_inmind() if a.bed == "inmind" else rsi.bed_prefeval()
    st = Store(bed)
    c0 = lm.STATS["calls"] + lm.STATS["cached"]
    use_d1 = "d1" in parts
    qt1 = None
    if use_d1:
        import methods as Mx
        lc = Mx.gen_cats_lines([l[:600] for l in st.lines])
        st.d1l = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in lc]

    # Dreamer + Analyst + Consolidator, all on the memory alone (no task requests are seen)
    probes = dream(st.lines)
    todo = [i for i, ps in enumerate(probes) if ps]
    noisy = Counter()
    for r in range(a.rounds):
        flat = [(i, p) for i in todo for p in probes[i]]
        if not flat:
            break
        qs = [p for _, p in flat]
        q1 = None
        if use_d1:
            import methods as Mx
            q1 = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in Mx.gen_cats_reqs(qs)]
        S = st.scores(qs, None, None, q1)
        missed = {}
        for (i, p), s in zip(flat, S):
            top = np.argsort(-s)[:K]
            if i not in top:
                missed.setdefault(i, []).append(p)
            if r == 0:
                for j in top:
                    if j != i:
                        noisy[int(j)] += 1
        print(f"  dream round {r}: probes {len(flat)} from {len(todo)} lines, buried lines {len(missed)}")
        ids = sorted(missed)
        notes = consolidate([(st.lines[i], missed[i]) for i in ids])
        for i, ns in zip(ids, notes):
            st.keys[i] += ns
        todo = ids
        if r + 1 < a.rounds:   # fresh probes for the lines that were still buried
            newp = dream([st.lines[i] for i in ids])
            for i, ps in zip(ids, newp):
                probes[i] = ps or probes[i]
    src = {i for i, ps in enumerate(probes) if ps}
    prune = np.array([1.0 if (noisy[j] >= 3 and j not in src) else 0.0 for j in range(len(st.lines))])
    print(f"  keys added to {sum(1 for k in st.keys if k)} lines, pruned (noisy) lines {int(prune.sum())}")

    base_scores = st.scores
    pen = [0.0]

    def scores(qs, ph=None, qt=None, q1=None):
        return [s - pen[0] * prune for s in base_scores(qs, ph, qt, q1)]
    st.scores = scores

    pidx = bridge.PropIndex(bridge.props_lines(st.lines)) if "prop" in parts else None
    dev = bed["stream"] + bed["gate"]
    best = None
    for p_ in (0.0, 0.02, 0.05):
        for w in ((0.25, 0.5, 1.0, 2.0) if pidx is not None else (0,)):
            pen[0] = p_
            d, o = bridge.score_set(st, dev, pidx, w, use_d1)
            sc = mean(d) + (mean(o) or 0)
            print(f"  dev pen={p_} w={w}: dense {mean(d)} off {mean(o)}")
            if best is None or sc > best[0]:
                best = (sc, p_, w)
    _, pen[0], w = best
    res = {}
    for name in ("testA", "testB"):
        if not bed[name]:
            res[name] = (None, None, [])
            continue
        d, o = bridge.score_set(st, bed[name], pidx, w, use_d1)
        res[name] = (mean(d), mean(o), o)
    dn, on = bridge.score_set(st, bed["testA"] + bed["testB"], pidx, w, use_d1, naive=True)
    calls = lm.STATS["calls"] + lm.STATS["cached"] - c0
    nB = len(bed["testB"])
    hits = int(sum(res["testB"][2])) if res["testB"][2] else int(round(res["testB"][0] * nB))
    print(f"[memdream {a.design}] pen={pen[0]} w={w} A off {res['testA'][1]} B off {res['testB'][1]} ({hits}/{nB}) "
          f"B dense {res['testB'][0]} ({int(round(res['testB'][0] * nB))}/{nB}) naive {mean(on)} calls {calls}")
    log(dict(kind="round", bed=a.bed, design="memdream:" + a.design, round=a.rounds, w=w, pen=pen[0],
             testA_off=res["testA"][1], testA_dense=res["testA"][0], testB_off=res["testB"][1],
             testB_dense=res["testB"][0], testB_per=res["testB"][2], naive_off=mean(on), calls_total=calls,
             rules=[], ntags=0, note="memdream.py"))


if __name__ == "__main__":
    main()
