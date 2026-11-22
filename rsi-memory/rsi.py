"""RSI architecture search for external memory on indirect requests.

  python rsi.py <design> [--bed inmind|prefeval] [--rounds 4]

Bed "inmind": InMind 2607.24368. 240 background user turns (fixed LME-s trace) + the 125 task facts.
  official score: per task, log = background + its one fact (241 lines, the paper / client setting).
  dense score:    one log with all 125 facts + background (365 lines), each request must find its fact.
  harm check:     recall of the task's direct ("naive") query in the official log.
Splits (by domain, never by wording): domains A = learning domains, B = unseen domains.
  A tasks split by hash into stream (24, fed in rounds of 6), gate (16, accept/reject), testA (24 unseen tasks).
  B tasks (61) = unseen-domain test.  Bed "prefeval": PrefEval log 4 = stream+gate, log 5 = test.
Every design returns, per round, recall@6 on testA, testB (official and dense) and naive recall.
No prompt below contains InMind wording, domain names or task examples.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import lm
from ret import embed, save

HERE = os.path.dirname(os.path.abspath(__file__))
K = 6
A_DOM = {"health_and_wellness", "legal", "consumer", "personal_development", "other"}


def h(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


# ---------------- beds ----------------

def bed_inmind():
    T = [json.loads(l) for l in open(os.path.join(HERE, "inmind_data/benchmark/dataset/inmind.jsonl"))]
    B = [json.loads(l) for l in open(os.path.join(HERE, "inmind_data/evaluation/background/lme_s_background.jsonl"))]
    bg = [re.sub(r"\n---TIMESTAMP:.*$", "", b["content"], flags=re.S).replace("\n", " ").strip() for b in B if b["role"] == "user"]
    lines = bg + [t["user_message"] for t in T]
    nb = len(bg)
    tasks = [dict(q=t["query"], naive=t["naive_query"], target=nb + i, dom=t["domain"], id=t["task_id"],
                  fact=t["user_message"]) for i, t in enumerate(T)]
    A = sorted([t for t in tasks if t["dom"] in A_DOM], key=lambda t: h(t["id"]))
    Bt = [t for t in tasks if t["dom"] not in A_DOM]
    return dict(name="inmind", lines=lines, nb=nb, stream=A[:24], gate=A[24:40], testA=A[40:], testB=Bt)


def bed_prefeval():
    L = [json.loads(l) for l in open(os.path.join(HERE, "..", "external-memory", "bed", "prefeval.jsonl"))]
    a, b = L[4], L[5]
    lines = a["lines"] + b["lines"]
    off = len(a["lines"])
    ta = sorted([dict(q=t["q"], naive=t["pref"], target=t["target"][0], id=f"a{i}", fact=a["lines"][t["target"][0]])
                 for i, t in enumerate(a["tasks"])], key=lambda t: h(t["id"]))
    tb = [dict(q=t["q"], naive=t["pref"], target=off + t["target"][0], id=f"b{i}", fact=b["lines"][t["target"][0]])
          for i, t in enumerate(b["tasks"])]
    # one store holding both logs (90 facts, 478 lines); learn on log-4 requests, score on log-5 requests
    return dict(name="prefeval", lines=lines, nb=None, stream=ta[:24], gate=ta[24:], testA=[], testB=tb)


# ---------------- model helpers ----------------

def jcall(prompt, system="You are a careful assistant. Reply with JSON only."):
    for att in range(2):
        out = lm.call(prompt + ("" if att == 0 else "\n\nReturn valid JSON only."), system)
        m = re.search(r"[\[{].*[\]}]", out, re.S)
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def by_id(j, n, field="q"):
    d = {}
    # Local models may return one ID-bearing object instead of a one-element array.
    rows = [j] if isinstance(j, dict) and "id" in j else (j if isinstance(j, list) else [])
    for r in rows:
        try:
            d[int(r["id"])] = r.get(field)
        except Exception:
            pass
    return [d.get(i) for i in range(n)]


def batched(items, size, fn):
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    res = lm.pmap(fn, chunks, workers=2)
    return [v for r in res for v in r]


def numbered(xs):
    return "\n".join(f"ID {i}: {x}" for i, x in enumerate(xs))


# ---------- read side: rule-guided search phrases ----------
EXP = ("A chat assistant keeps a memory of lines the user wrote in earlier conversations. For each new request "
       "below, write 3 short search phrases, each describing a kind of personal fact the user may have stated "
       "earlier (a condition, limit, preference, family member, possession, belief, job or situation) that should "
       "change how a careful assistant answers this request.{rules}\nReply with a JSON list of objects "
       "{{\"id\": <id>, \"q\": [<3 phrases>]}}.\n\nREQUESTS:\n{reqs}")


def rules_block(rules):
    if not rules:
        return ""
    return "\nFollow these search rules, learned from earlier misses, when they apply:\n" + "\n".join(f"- {r}" for r in rules)


def expand(qs, rules):
    rb = rules_block(rules)

    def go(ch):
        j = jcall(EXP.format(rules=rb, reqs=numbered(ch)))
        return [[x for x in (v or []) if isinstance(x, str)] for v in by_id(j, len(ch))]
    return batched(list(qs), 12, go)


# ---------- write side: implication keys ----------
IMP0 = "For each line, think of the everyday tasks where this fact about the user would change a careful answer."
IMPP = ("Below are lines a user wrote to a chat assistant. {instr}\nFor each line that states a lasting fact about "
        "the user, write 3 short requests the user might make later, each on a different everyday task, whose good "
        "answer would change because of this fact (the requests must not mention the fact). For a line with no "
        "lasting fact, give an empty list.\nReply with a JSON list of objects {{\"id\": <id>, \"q\": [<requests>]}}."
        "\n\nLINES:\n{lines}")


def imp_keys(lines, instr):
    def go(ch):
        j = jcall(IMPP.format(instr=instr, lines=numbered([l[:600] for l in ch])))
        return [[x for x in (v or []) if isinstance(x, str)][:3] for v in by_id(j, len(ch))]
    return batched(list(lines), 30, go)


# ---------- tags (self-grown vocabulary) ----------
TAGL = ("Below are concern tags, then lines a user wrote to a chat assistant. For each line, list the tag numbers "
        "whose definition the line matches (a fact about the user of that kind). Most lines match none.\n"
        "Reply with a JSON list of objects {{\"id\": <id>, \"t\": [<numbers>]}}.\n\nTAGS:\n{tags}\n\nLINES:\n{lines}")
TAGQ = ("Below are concern tags, then requests a user made to a chat assistant. For each request, list the tag "
        "numbers whose kind of user fact could change a careful answer to it (usually 0-3).\nReply with a JSON list "
        "of objects {{\"id\": <id>, \"t\": [<numbers>]}}.\n\nTAGS:\n{tags}\n\nREQUESTS:\n{reqs}")


def tag_list(tags):
    return "\n".join(f"{i}. {n}: {d}" for i, (n, d) in enumerate(tags))


def tag_lines(lines, tags):
    def go(ch):
        j = jcall(TAGL.format(tags=tag_list(tags), lines=numbered([l[:400] for l in ch])))
        return [set(int(x) for x in (v or []) if str(x).isdigit() and int(x) < len(tags)) for v in by_id(j, len(ch), "t")]
    return batched(list(lines), 40, go)


def tag_reqs(qs, tags):
    def go(ch):
        j = jcall(TAGQ.format(tags=tag_list(tags), reqs=numbered(ch)))
        return [set(int(x) for x in (v or []) if str(x).isdigit() and int(x) < len(tags)) for v in by_id(j, len(ch), "t")]
    return batched(list(qs), 20, go)


# ---------------- scoring ----------------

class Store:
    def __init__(self, bed):
        self.bed = bed
        self.lines = bed["lines"]
        self.L = embed(self.lines)
        self.keys = [[] for _ in self.lines]   # extra key texts per line
        self.ltags = None                      # list of sets
        self.tags = []
        self.d1l = None                        # fixed 50-activity labels per line (D1), if used
        self.w = 1.0

    def key_matrix(self):
        flat, owner = [], []
        for i, ks in enumerate(self.keys):
            for k in ks:
                flat.append(k)
                owner.append(i)
        return (embed(flat) if flat else None), np.array(owner, dtype=int)

    def scores(self, qs, phrases=None, qtags=None, qtags1=None):
        """line scores per request: max over query vectors (request, request+phrase mixes) of max over keys"""
        Q = embed(list(qs), query=True)
        KM, own = self.key_matrix()
        out = []
        for i, qv in enumerate(Q):
            vs = [qv]
            if phrases and phrases[i]:
                P = embed(phrases[i], query=True)
                for pv in P:
                    m = pv + qv
                    vs.append(m / np.linalg.norm(m))
            V = np.stack(vs)
            s = (self.L @ V.T).max(1)
            if KM is not None:
                ks = (KM @ V.T).max(1)
                np.maximum.at(s, own, ks)
            for qt_, lt_, n_ in ((qtags, self.ltags, len(self.tags)), (qtags1, self.d1l, 50)):
                if qt_ is not None and lt_ is not None and qt_[i]:
                    df = np.bincount([t for ts in lt_ for t in ts], minlength=n_) + 1
                    bonus = np.array([sum(1.0 / np.log(df[t] + 1) for t in (qt_[i] & lt)) for lt in lt_])
                    s = s + self.w * bonus
            out.append(s)
        return out

    def recall(self, tasks, S):
        """dense recall@6 and official recall@6 (background + own fact only)"""
        dense, off = [], []
        nb = self.bed["nb"]
        for t, s in zip(tasks, S):
            tg = t["target"]
            dense.append(float(tg in np.argsort(-s)[:K]))
            if nb is not None:
                off.append(float((s[:nb] > s[tg]).sum() < K))   # any target: rank among background lines only
        return dense, off


def evaluate(store, tasks, rules=None, use_exp=False, use_tags=False, naive=False):
    qs = [t["naive"] if naive else t["q"] for t in tasks]
    ph = expand(qs, rules) if use_exp else None
    qt = None
    qt1 = None
    if store.d1l is not None:
        import methods as Mx
        qt1 = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in Mx.gen_cats_reqs(qs)]
    if use_tags == "hyde":
        import methods as Mx
        ph = [[n] if n else [] for n in Mx.gen_hyde(qs)]
    elif use_tags and store.tags:
        qt = tag_reqs(qs, store.tags)
    S = store.scores(qs, ph, qt, qt1)
    d, o = store.recall(tasks, S)
    return d, o, S, ph


# ---------------- learning modules ----------------
RW_PROMPT = ("A memory search for a chat assistant missed stored facts it needed. Each case shows a request, the "
             "stored line that should have been found, and the search phrases that were used. Write general search "
             "rules that would make the search find such facts for OTHER requests of the same kind. A rule names a "
             "kind of request and the kinds of personal facts to also search for, for example: \"When a request is "
             "about planning a trip, also search for mobility limits, money limits and fears.\" Rules must be general: "
             "never quote the request or the line, never name specific products, people, places or brands.")
RW_TAIL = ("\n\nCURRENT RULES:\n{rules}\n\nCASES:\n{cases}\n\nWrite at most {n} new rules that are not already "
           "covered. Reply with a JSON list of strings.")


def write_rules(cases, rules, n=3, prompt=RW_PROMPT):
    cs = "\n\n".join(f"REQUEST: {q}\nNEEDED LINE: {f[:300]}\nPHRASES USED: {'; '.join(p or [])}" for q, f, p in cases)
    j = jcall(prompt + RW_TAIL.format(rules="\n".join(f"- {r}" for r in rules) or "(none)", cases=cs, n=n))
    return [r for r in (j if isinstance(j, list) else []) if isinstance(r, str)][:n]


MF = ("Below is everything a user told a chat assistant in earlier conversations (numbered lines), then new "
      "requests from the same user. For each request, list the numbers of at most 2 lines that state a fact about "
      "the user a careful assistant must take into account when answering it, or an empty list if none applies.\n"
      "Reply with a JSON list of objects {{\"id\": <id>, \"lines\": [<numbers>]}}.\n\nLINES:\n{lines}\n\nREQUESTS:\n{reqs}")


def miss_finder(store, qs):
    lines = "\n".join(f"{i}. {l[:240]}" for i, l in enumerate(store.lines))

    def go(ch):
        j = jcall(MF.format(lines=lines, reqs=numbered(ch)))
        return [[int(x) for x in (v or []) if str(x).isdigit()] for v in by_id(j, len(ch), "lines")]
    return batched(list(qs), 8, go)


PRACT = ("Below are lines a user wrote to a chat assistant. For each line that states a lasting fact about the user, "
         "write ONE request the user might make much later, on an everyday task, where a careful answer must take "
         "this fact into account but the request does not mention it. For other lines give null.\n"
         "Reply with a JSON list of objects {{\"id\": <id>, \"q\": <request or null>}}.\n\nLINES:\n{lines}")


def practice(lines):
    def go(ch):
        j = jcall(PRACT.format(lines=numbered([l[:600] for l in ch])))
        return [v if isinstance(v, str) else None for v in by_id(j, len(ch))]
    return batched(list(lines), 30, go)


META = ("Below is an instruction given to a model that writes search rules for a chat assistant's memory after the "
        "search missed a needed fact, then the rules it wrote and whether they helped on held-out requests. Rewrite "
        "the instruction so the rules it produces find more needed facts on new kinds of requests while staying "
        "general. Keep it under 150 words and keep the example format. Reply with JSON {{\"instruction\": <text>}}."
        "\n\nINSTRUCTION:\n{p}\n\nRULES IT WROTE:\n{rules}\n\nRESULT: {res}")

IMPED = ("Below is an instruction used when writing, for each stored fact about a user, short later requests whose "
         "answer the fact would change (these become search keys). Then cases where a later request still missed "
         "its fact, with the keys that had been written for that fact. Rewrite the instruction (under 80 words) so "
         "the keys written for ANY fact cover such requests better. Stay general: do not mention these facts, "
         "requests, products or places. Reply with JSON {{\"instruction\": <text>}}.\n\nINSTRUCTION:\n{p}\n\nCASES:\n{cases}")

TAGP = ("A memory search for a chat assistant missed stored facts. Each case shows a request and the stored line "
        "that should have been found. Propose concern tags that link kinds of requests to kinds of user facts, so "
        "both the line and the request would get the same tag. A tag has a short name and a one-line definition of "
        "the kind of user fact it covers and the kinds of requests it matters for. Tags must be general (never name "
        "specific products, people or places) and must not duplicate existing tags.\n\nEXISTING TAGS:\n{tags}\n\n"
        "CASES:\n{cases}\n\nPropose at most {n} new tags. Reply with a JSON list of objects {{\"name\": <name>, \"def\": <definition>}}.")

BAD_RULES = ["When a request mentions any purchase, also search for the user's favourite colours and music.",
             "For every request, also search for what the user ate yesterday.",
             "When a request is about work, also search for the user's hobbies and sports teams."]


def log(row):
    row["t"] = time.strftime("%F %T")
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def mean(x):
    return round(float(np.mean(x)), 4) if len(x) else None


def run(design, bedname, rounds, tag=""):
    bed = bed_inmind() if bedname == "inmind" else bed_prefeval()
    st = Store(bed)
    c0 = lm.STATS["calls"] + lm.STATS["cached"]
    parts = set(design.split("+"))
    use_exp = bool(parts & {"exp", "rules", "rulewriter", "missfinder", "selfrewrite", "critic", "badrules"})
    use_tags = "tags" in parts
    if "hyde" in parts:
        use_tags = "hyde"
    rules = []
    if "rules" in parts:  # static hand-written generic rule list (C1)
        rules = ["When a request asks for a recommendation or a purchase, also search for health conditions, "
                 "allergies, diets, children or pets at home, and money limits.",
                 "When a request is about an activity, trip or event, also search for physical limits, medicines, "
                 "beliefs and commitments that could rule it out.",
                 "When a request asks whether something is a good idea, also search for the user's job, legal or "
                 "family situation that changes the risks.",
                 "When a request is about food, drink or cooking, also search for diets, religion and medical limits."]
    if "badrules" in parts:
        rules = rules + list(BAD_RULES)
    instr = IMP0
    rw_prompt = RW_PROMPT
    if "impkeys" in parts:
        ks = imp_keys(st.lines, instr)
        st.keys = [list(k or []) for k in ks]
    if "d2q" in parts:   # baseline: fixed doc2query-style keys (3 per line), never updated
        st.keys = [list(k or []) for k in imp_keys(st.lines, IMP0)]
    if parts & {"hyde", "d1", "d3"}:
        sys.path.append(os.path.join(HERE, "..", "external-memory"))
        import methods as Mx   # its prompts; calls go through this folder's lm (own cache)
    if "d3" in parts:    # baseline: one earlier, different request per fact; added as a key when its search missed
        allt = bed["stream"] + bed["gate"] + bed["testA"] + bed["testB"]
        early = Mx.gen_early([dict(pref=t["fact"], q=t["q"]) for t in allt])
        S = st.scores([e or "" for e in early])
        for t, e, s in zip(allt, early, S):
            if e and t["target"] not in np.argsort(-s)[:K]:
                st.keys[t["target"]].append(e)
    if "d1" in parts:    # baseline: fixed list of 50 activity categories on both sides, rarity-weighted overlap
        lc = Mx.gen_cats_lines([l[:600] for l in st.lines])
        st.d1l = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in lc]
    if "freetags" in parts:   # control: same number of tags (8), written once with no misses and no gate
        j = jcall("Propose 8 concern tags for the memory of a personal chat assistant. Each tag links a kind of fact "
                  "about the user to the kinds of later requests where that fact should change a careful answer. "
                  "A tag has a short name and a one-line definition. Tags must be general. Reply with a JSON list of "
                  "objects {\"name\": <name>, \"def\": <definition>}.")
        st.tags = [(x["name"], x["def"]) for x in (j if isinstance(j, list) else []) if isinstance(x, dict) and "name" in x][:8]
        st.ltags = tag_lines(st.lines, st.tags)
        use_tags = True
    hist = []
    stream = bed["stream"]
    per = max(1, len(stream) // rounds)
    pr_stream = pr_gate = None
    if "practice" in parts:   # self-made requests replace the user stream (no gold labels from users)
        pq = practice(st.lines)
        pr = [dict(q=q, target=i, naive="", id=f"p{i}") for i, q in enumerate(pq) if q]
        pr.sort(key=lambda t: h(t["id"]))
        pr_stream, pr_gate = pr[:8 * rounds], pr[8 * rounds:8 * rounds + 24]
        per = 8
    mf_stats = []

    def score_round(r, note=""):
        out = dict(design=design + tag, bed=bedname, round=r, rules=list(rules), ntags=len(st.tags), instr=instr, note=note)
        for name in ("testA", "testB"):
            ts = bed[name]
            if not ts:
                continue
            d, o, _, _ = evaluate(st, ts, rules, use_exp, use_tags)
            out[name + "_dense"] = mean(d)
            out[name + "_off"] = mean(o)
            out[name + "_per"] = d if not o else o
        if bed["nb"] is not None:  # harm check: direct question recall, official log
            ts = bed["testA"] + bed["testB"]
            d, o, _, _ = evaluate(st, ts, rules, use_exp, use_tags, naive=True)
            out["naive_off"] = mean(o)
        out["calls_total"] = lm.STATS["calls"] + lm.STATS["cached"] - c0
        log(dict(kind="round", **out))
        hist.append(out)
        print(f"[{design}{tag}] r{r} A {out.get('testA_off')}/{out.get('testA_dense')} "
              f"B {out.get('testB_off')}/{out.get('testB_dense')} naive {out.get('naive_off')} "
              f"rules {len(rules)} tags {len(st.tags)} calls {out['calls_total']}", flush=True)

    def gate_score(rl=None, use=None):
        g = pr_gate if pr_gate is not None else bed["gate"]
        d, o, _, _ = evaluate(st, g, rules if rl is None else rl, use_exp, use_tags)
        return round(mean(d) + (mean(o) or 0), 4)

    score_round(0)
    learning = parts & {"rulewriter", "missfinder", "practice", "selfrewrite", "critic", "impkeys", "tags", "badrules"}
    if not learning:
        return hist
    for r in range(1, rounds + 1):
        chunk = (pr_stream if pr_stream is not None else stream)[(r - 1) * per: r * per]
        d, o, S, ph = evaluate(st, chunk, rules, use_exp, use_tags)
        top = [list(np.argsort(-s)[:K]) for s in S]
        if "missfinder" in parts:   # no gold: an offline scan of the whole store flags what should have been found
            flags = miss_finder(st, [t["q"] for t in chunk])
            cases = []
            for t, f, tp, p in zip(chunk, flags, top, ph or [None] * len(chunk)):
                gold_missed = t["target"] not in tp
                fl = [i for i in f if 0 <= i < len(st.lines) and i not in tp]
                mf_stats.append(dict(gold_missed=gold_missed, flagged=bool(fl), right=t["target"] in f))
                for i in fl[:1]:
                    cases.append((t["q"], st.lines[i], p))
        else:
            cases = [(t["q"], st.lines[t["target"]], p) for t, tp, p in zip(chunk, top, ph or [None] * len(chunk))
                     if t["target"] not in tp]
        note = f"misses {len(cases)}/{len(chunk)}"
        if cases and parts & {"rulewriter", "missfinder", "selfrewrite"}:
            g0 = gate_score()
            new = write_rules(cases, rules, 3, rw_prompt)
            g1 = gate_score(rules + new)
            keep = g1 > g0
            if "selfrewrite" in parts:
                j = jcall(META.format(p=rw_prompt, rules="\n".join(new), res=f"gate recall {g0} -> {g1}"))
                p2 = (j or {}).get("instruction") if isinstance(j, dict) else None
                if p2:
                    new2 = write_rules(cases, rules, 3, p2)
                    g2 = gate_score(rules + new2)
                    if g2 > max(g1, g0):
                        rw_prompt, new, g1, keep = p2, new2, g2, True
                        note += " prompt-rewritten"
            if keep:
                rules = rules + new
            note += f" gate {g0}->{g1} {'keep' if keep else 'reject'}"
        if "critic" in parts and rules:  # leave-one-out credit: drop rules whose removal does not lower gate recall
            base = gate_score()
            kept = []
            for i, rr in enumerate(rules):
                without = [x for j2, x in enumerate(rules) if j2 != i]
                if gate_score(without) >= base:
                    note += f" drop[{rr[:40]}]"
                else:
                    kept.append(rr)
            rules = kept
        if "impkeys" in parts and cases:
            cs = "\n\n".join(f"LATER REQUEST: {q}\nFACT: {f[:300]}\nKEYS WRITTEN: {'; '.join(st.keys[st.lines.index(f)] if f in st.lines else [])}"
                             for q, f, _ in cases)
            j = jcall(IMPED.format(p=instr, cases=cs))
            ni = (j or {}).get("instruction") if isinstance(j, dict) else None
            if ni:
                g0 = gate_score()
                old = st.keys
                st.keys = [list(k or []) for k in imp_keys(st.lines, ni)]
                g1 = gate_score()
                if g1 > g0:
                    instr = ni
                    note += f" instr gate {g0}->{g1} keep"
                else:
                    st.keys = old
                    note += f" instr gate {g0}->{g1} reject"
        if "tags" in parts and cases:
            cs = "\n\n".join(f"REQUEST: {q}\nNEEDED LINE: {f[:300]}" for q, f, _ in cases)
            j = jcall(TAGP.format(tags=tag_list(st.tags) or "(none)", cases=cs, n=3))
            nt = [(x["name"], x["def"]) for x in (j if isinstance(j, list) else []) if isinstance(x, dict) and "name" in x and "def" in x][:3]
            if nt:
                g0 = gate_score()
                old_t, old_l = st.tags, st.ltags
                st.tags = st.tags + nt
                st.ltags = tag_lines(st.lines, st.tags)
                # flood check: a tag on more than 15% of lines is dropped (split is left to the next round's proposal)
                df = np.bincount([t for ts in st.ltags for t in ts], minlength=len(st.tags))
                flood = [i for i in range(len(st.tags)) if df[i] > 0.15 * len(st.lines)]
                if flood:
                    note += f" flood-drop {[st.tags[i][0] for i in flood]}"
                    keepi = [i for i in range(len(st.tags)) if i not in flood]
                    remap = {o: n for n, o in enumerate(keepi)}
                    st.tags = [st.tags[i] for i in keepi]
                    st.ltags = [set(remap[t] for t in ts if t in remap) for ts in st.ltags]
                g1 = gate_score()
                if g1 > g0 or "nogate" in parts:
                    note += f" tags gate {g0}->{g1} keep {[n for n, _ in nt]}"
                else:
                    st.tags, st.ltags = old_t, old_l
                    note += f" tags gate {g0}->{g1} reject"
        score_round(r, note)
    if mf_stats:
        gm = [m for m in mf_stats if m["gold_missed"]]
        log(dict(kind="missfinder", design=design + tag, bed=bedname, n=len(mf_stats),
                 flagged_when_missed=mean([m["flagged"] for m in gm]),
                 right_line=mean([m["right"] for m in mf_stats]),
                 false_flags=mean([m["flagged"] for m in mf_stats if not m["gold_missed"]])))
    save()
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--bed", default="inmind")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    run(a.design, a.bed, a.rounds, a.tag)
    save()
    print("lm", lm.STATS, f"{time.time() - t0:.0f}s")
