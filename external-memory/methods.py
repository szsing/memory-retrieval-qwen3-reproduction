"""Memory methods. run(name, logs, slice) returns, per log, per task, the 6 line ids handed to the model.

Published baselines (each with its main knob):
  plain           request embedded as the query, each stored line one key
  <method>/s      suffix: each sentence of a line is its own key (line score = best sentence)    [chunk knob]
  rerank:N        plain_sent top N, then a cross-encoder (MiniLM, MS MARCO) rescoring              [N]
  hyde:MODE       model writes a made-up earlier user note that would matter; MODE=only|mix (mean with request)
  d2q:N           doc2query: model writes N future requests per stored line, added as keys      [N]
  mem0:MODE       Mem0-style: model extracts facts per chunk of 15 lines; search facts, hand over source lines;
                  MODE=facts (facts only) | both (facts and raw lines as keys)
  amem:MODE       A-MEM-style note per line (keywords, tags, context); MODE=concat | max
  regen:MODE      ITER-RETGEN one round: answer with plain top 6, search again with MODE=draft|both
New designs:
  d1:R            closed activity-category routing (CATS); R slots reserved for category matches
  d2:MODE         draft-then-conflict: model drafts concrete suggestions with no memory; each item is a query
                  MODE=items (max over items) | mix (items and request)
Controls: none (no notes), ceiling (target lines handed over).
"""
import json
import re

import numpy as np

import lm
from ret import cross, embed

K = int(__import__("os").environ.get("MEM_K", 6))
SENT = re.compile(r"(?<=[.!?])\s+")

CATS = ["meals and snacks", "restaurants and cafes", "cooking and recipes", "drinks and alcohol", "grocery shopping",
        "hotels and lodging", "flights and air travel", "road trips and car rental", "public transport and commuting",
        "choosing travel destinations", "outdoor activities and nature", "beach, pool and water activities",
        "playing sports", "watching sports", "gym and workouts", "health and medical care", "mental health and stress",
        "sleep", "beauty and skincare", "clothing and fashion", "home furniture and decor", "home repair and DIY",
        "gardening", "pets and animals", "kids and family activities", "dating and relationships",
        "gifts and celebrations", "parties and social events", "music", "books and reading", "movies and TV shows",
        "video games and board games", "arts and crafts", "taking courses and classes", "how I learn and study",
        "work and career", "where and how I work", "money and budgeting", "buying cars and big appliances",
        "phones, computers and gadgets", "software and apps", "driving and vehicles", "environment and sustainability",
        "religion and personal values", "accessibility and mobility", "weather and climate", "noise and crowds",
        "daily schedule and time of day", "languages and communication", "photography and media"]


# ---------- helpers ----------

def sents(line):
    s = [x for x in SENT.split(line) if len(x) > 3]
    return s or [line]


def key_scores(qvs, keys_per_line, agg_q="max"):
    """Line score = max over its keys of (max or mean over query vectors) cosine."""
    flat, owner = [], []
    for i, ks in enumerate(keys_per_line):
        flat += ks
        owner += [i] * len(ks)
    E = embed(flat)
    S = E @ qvs.T  # keys x queries
    s = S.max(1) if agg_q == "max" else S.mean(1)
    out = np.full(len(keys_per_line), -9.0)
    np.maximum.at(out, np.array(owner), s)
    return out


def top(scores, k=K):
    return [int(i) for i in np.argsort(-scores)[:k]]


def batched(items, size, make_prompt, parse, system="You are a careful assistant. Reply with JSON only."):
    """Ask the model about many items per call; returns one parsed value per item."""
    chunks = [items[i:i + size] for i in range(0, len(items), size)]

    def go(ch):
        for attempt in range(2):
            out = lm.call(make_prompt(ch) + ("" if attempt == 0 else "\n\nReturn valid JSON only."), system)
            try:
                m = re.search(r"[\[{].*[\]}]", out, re.S)
                return parse(json.loads(m.group(0)), ch)
            except Exception:
                continue
        return [None] * len(ch)

    res = lm.pmap(go, chunks, workers=2)
    return [v for r in res for v in r]


def by_id(j, ch, field):
    if isinstance(j, dict):
        j = j.get("items", list(j.values())[0] if len(j) == 1 else [])
    d = {}
    for r in j:
        try:
            d[int(r["id"])] = r.get(field)
        except Exception:
            pass
    return [d.get(i) for i, _ in ch]


def numbered(ch):
    return "\n".join(f"ID {i}: {t}" for i, t in ch)


# ---------- model-written parts (cached by lm) ----------

def gen_d2q(lines, n):
    p = lambda ch: (f"Below are lines a user wrote to a chat assistant. For each line, write {n} different requests "
                    f"the same user might make weeks later where this line would help the assistant answer well. "
                    f"Reply with a JSON list of objects {{\"id\": <id>, \"q\": [<{n} requests>]}}.\n\n{numbered(ch)}")
    return batched(list(enumerate(lines)), 30, p, lambda j, ch: by_id(j, ch, "q"))


def gen_hyde(qs):
    p = lambda ch: ("For each request below, imagine earlier conversations with the same user. Write one or two "
                    "sentences that the user may have said earlier about themselves that would change how a good "
                    "assistant answers the request. Reply with a JSON list of objects {\"id\": <id>, \"note\": <text>}."
                    f"\n\n{numbered(ch)}")
    return batched(list(enumerate(qs)), 15, p, lambda j, ch: by_id(j, ch, "note"))


def gen_facts(lines):
    """Mem0-style extraction over chunks of 15 consecutive lines; facts keep their source line id."""
    chunks = [(s, lines[s:s + 15]) for s in range(0, len(lines), 15)]
    p = lambda c: ("Extract the facts about the user that are worth remembering for future conversations "
                   "(preferences, dislikes, constraints, health, habits, plans, people, possessions) from the "
                   "user's messages below. Each fact is one short sentence and names the line it came from. "
                   "Reply with a JSON list of objects {\"line\": <line id>, \"fact\": <text>}.\n\n"
                   + "\n".join(f"LINE {c[0] + i}: {t}" for i, t in enumerate(c[1])))

    def parse(j, ch):
        if isinstance(j, dict):
            j = list(j.values())[0]
        return [[(int(r["line"]), r["fact"]) for r in j if isinstance(r, dict) and "line" in r and "fact" in r]]

    res = batched(chunks, 1, lambda ch: p(ch[0]), parse)
    return [f for r in res for f in (r or [])]


def gen_amem(lines):
    p = lambda ch: ("For each user message below, write a memory note as A-MEM does: keywords (up to 5), tags "
                    "(up to 4 broad categories) and a one-sentence context describing what it is about. "
                    "Reply with a JSON list of objects {\"id\": <id>, \"keywords\": [...], \"tags\": [...], "
                    f"\"context\": <text>}}.\n\n{numbered(ch)}")

    def parse(j, ch):
        d = {}
        for r in (j if isinstance(j, list) else []):
            try:
                d[int(r["id"])] = "keywords: " + ", ".join(r.get("keywords", [])) + ". tags: " + \
                                  ", ".join(r.get("tags", [])) + ". " + r.get("context", "")
            except Exception:
                pass
        return [d.get(i) for i, _ in ch]

    return batched(list(enumerate(lines)), 30, p, parse)


G9 = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 44],
      [19, 20, 21, 22, 38, 39, 40, 41], [23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 49],
      [33, 34, 35, 36, 37, 47, 48], [42, 43, 45, 46]]
G5 = [G9[0] + G9[1], G9[2] + G9[3], G9[4] + G9[8], G9[5] + G9[6], G9[7]]  # CreaMem-style: five coarse scenes


def coarsen(cats, groups):
    m = {c: g for g, cs in enumerate(groups) for c in cs}
    return set(m[c] for c in cats if c in m)


CATLIST = "\n".join(f"{i}. {c}" for i, c in enumerate(CATS))


def gen_cats_lines(lines):
    p = lambda ch: ("Below is a fixed list of activity categories, then lines a user wrote to a chat assistant.\n"
                    "For each line: if it states a lasting fact about the user (a preference, dislike, constraint, "
                    "health condition, habit, possession, family member, plan), list the numbers of EVERY category "
                    "where this fact should change a good recommendation or answer, including indirect ones "
                    "(a peanut allergy matters for meals, restaurants, flights, gifts, cooking). If the line states "
                    "no lasting fact, give an empty list.\nReply with a JSON list of objects {\"id\": <id>, "
                    f"\"cats\": [<numbers>]}}.\n\nCATEGORIES:\n{CATLIST}\n\nLINES:\n{numbered(ch)}")
    return batched(list(enumerate(lines)), 30, p, lambda j, ch: by_id(j, ch, "cats"))


def gen_cats_reqs(qs):
    p = lambda ch: ("Below is a fixed list of activity categories, then requests a user made to a chat assistant.\n"
                    "For each request, list the numbers of the categories (usually 1-4) whose facts about the user "
                    "could change a good answer to it.\nReply with a JSON list of objects {\"id\": <id>, \"cats\": "
                    f"[<numbers>]}}.\n\nCATEGORIES:\n{CATLIST}\n\nREQUESTS:\n{numbered(ch)}")
    return batched(list(enumerate(qs)), 20, p, lambda j, ch: by_id(j, ch, "cats"))


def gen_drafts(qs):
    p = lambda ch: ("For each request below, you know nothing about the user. List 5 concrete things you would "
                    "suggest or include in your answer (specific items, places, foods, products, activities), "
                    "each as a short phrase. Reply with a JSON list of objects {\"id\": <id>, \"items\": [<5 phrases>]}."
                    f"\n\n{numbered(ch)}")
    return batched(list(enumerate(qs)), 15, p, lambda j, ch: by_id(j, ch, "items"))


def gen_early(tasks):
    """D3 bed: one earlier request per fact, a different task where the same fact matters (never names it)."""
    items = [(i, f"PREFERENCE: {t['pref']} | LATER REQUEST (do not reuse its topic): {t['q']}") for i, t in enumerate(tasks)]
    p = lambda ch: ("Each item gives a user's preference and one request they will make later. Write one DIFFERENT "
                    "request the same user could make earlier, on a different task or topic, where a good answer "
                    "must also respect the preference. The request must not mention or hint at the preference "
                    "itself. Reply with a JSON list of objects {\"id\": <id>, \"q\": <request>}.\n\n" + numbered(ch))
    return batched(items, 15, p, lambda j, ch: by_id(j, ch, "q"))


def gen_type(lines):
    """D4 typed store: is the line a standing constraint or preference, an event, or neither."""
    p = lambda ch: ("For each line a user wrote to a chat assistant, give its type: \"standing\" if it states a "
                    "lasting preference, dislike, constraint, health condition or habit of the user that should "
                    "shape future recommendations; \"event\" if it reports something that happened or a one-time "
                    "plan; \"other\" otherwise (questions, chit-chat, facts about the world). Reply with a JSON list "
                    "of objects {\"id\": <id>, \"type\": <standing|event|other>}.\n\n" + numbered(ch))
    return batched(list(enumerate(lines)), 30, p, lambda j, ch: by_id(j, ch, "type"))


REGEN_SYS = ("You are a helpful assistant that remembers earlier conversations with this user. Notes retrieved from "
             "memory are given; some may be irrelevant. Answer in at most 80 words.")


def gen_regen(pairs):
    """pairs: (request, notes) -> first-round answer, one call per request."""
    def go(x):
        q, notes = x
        return lm.call("NOTES FROM MEMORY:\n" + "\n".join("- " + n for n in notes) + f"\n\nUSER REQUEST: {q}", REGEN_SYS)
    return lm.pmap(go, pairs, workers=2)


RANK_SYS = "You pick memory notes for an assistant. Reply with JSON only."


def gen_llmrank(pairs):
    """pairs: (request, candidate lines) -> ordered list of picked candidate positions (listwise LLM rerank,
    RankGPT-style), one call per request."""
    def go(x):
        q, cand = x
        p = ("Below are numbered lines a user wrote to a chat assistant in earlier conversations, then a new request "
             "from the same user. Pick the lines (at most 6) that state a fact about the user (a preference, dislike, "
             "constraint, health condition, habit) which a good answer to the request should take into account, "
             "including indirect ones. Most important first. Reply with a JSON list of line numbers.\n\nLINES:\n"
             + "\n".join(f"{i}. {c}" for i, c in enumerate(cand)) + f"\n\nREQUEST: {q}")
        out = lm.call(p, RANK_SYS)
        m = re.search(r"\[[^\]]*\]", out)
        try:
            return [int(v) for v in json.loads(m.group(0)) if 0 <= int(v) < len(cand)]
        except Exception:
            return []
    return lm.pmap(go, pairs, workers=2)


# ---------- methods ----------

# ---------- derived logs (scale grid): generate on each source log, then map ----------
_SRC, _MEMO = {}, {}


def _src(name):
    if not _SRC:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("prefeval.jsonl", "locomo.jsonl"):
            for l in open(os.path.join(here, "bed", f)):
                x = json.loads(l)
                _SRC[x["log"]] = x
    return _SRC[name]


def LG(fn, L, lines, *a):
    """per-line generator on a derived log: run fn on each source log's full lines (cached calls), then map"""
    if "src" not in L:
        return fn(lines, *a)
    res = {}
    for nm in dict.fromkeys(n for n, _ in L["src"]):
        k = (fn.__name__, nm, a)
        if k not in _MEMO:
            _MEMO[k] = fn(_src(nm)["lines"], *a)
        res[nm] = _MEMO[k]
    return [res[n][j] for n, j in L["src"]]


def QG(fn, L, qs, *a):
    """per-request generator on a derived log, same idea over the source logs' tasks"""
    if "tsrc" not in L:
        return fn(qs, *a)
    res = {}
    for nm in dict.fromkeys(n for n, _ in L["tsrc"]):
        k = (fn.__name__, nm, a)
        if k not in _MEMO:
            _MEMO[k] = fn([t["q"] for t in _src(nm)["tasks"]], *a)
        res[nm] = _MEMO[k]
    return [res[n][j] for n, j in L["tsrc"]]


def run(name, logs, slice_):
    sentkeys = name.endswith("/s")
    base, _, arg = name.removesuffix("/s").partition(":")
    out = []
    for L in logs:
        lines, tasks = L["lines"], L["tasks"]
        qs = [t["q"] for t in tasks]
        Q = embed(qs, query=True)
        skeys = [sents(l) for l in lines] if sentkeys else [[l] for l in lines]
        if base == "none":
            out.append([[] for _ in tasks])
            continue
        if base == "ceiling":
            out.append([t["target"][:K] for t in tasks])
            continue
        plain_s = [key_scores(q[None], skeys) for q in Q]
        if base == "plain":
            out.append([top(s) for s in plain_s])
        elif base == "rerank":
            n = int(arg or 50)
            res = []
            for q, s in zip(qs, plain_s):
                cand = top(s, n)
                cs = cross([(q, lines[i]) for i in cand])
                res.append([cand[i] for i in np.argsort(-cs)[:K]])
            out.append(res)
        elif base == "hyde":
            notes = QG(gen_hyde, L, qs)
            res = []
            for q, qv, nt, s in zip(qs, Q, notes, plain_s):
                if not nt:
                    res.append(top(s))
                    continue
                hv = embed([nt])[0]
                v = hv if arg == "only" else (hv + qv) / np.linalg.norm(hv + qv)
                res.append(top(key_scores(v[None], skeys)))
            out.append(res)
        elif base == "d2q":
            n = int(arg or 1)
            gq = LG(gen_d2q, L, lines, n)
            keys = [ks + [x for x in (g or [])[:n] if isinstance(x, str)] for ks, g in zip(skeys, gq)]
            out.append([top(key_scores(q[None], keys)) for q in Q])
        elif base == "d2qhyde":
            # equal-cost baseline for D1: write-time doc2query keys (one call per batch of lines) plus a
            # read-time HyDE note (one call per request), the same call budget D1 uses for its two tag lists
            n = int(arg or 3)
            gq = LG(gen_d2q, L, lines, n)
            keys = [ks + [x for x in (g or [])[:n] if isinstance(x, str)] for ks, g in zip(skeys, gq)]
            notes = QG(gen_hyde, L, qs)
            res = []
            for qv, nt in zip(Q, notes):
                v = qv
                if nt:
                    hv = embed([nt])[0]
                    v = (hv + qv) / np.linalg.norm(hv + qv)
                res.append(top(key_scores(v[None], keys)))
            out.append(res)
        elif base == "mem0":
            if "src" in L:  # derived log: reuse the facts written for each source log
                pos = {s: i for i, s in enumerate(map(tuple, L["src"]))}
                facts = []
                for nm in dict.fromkeys(n for n, _ in L["src"]):
                    k = ("gen_facts", nm)
                    if k not in _MEMO:
                        _MEMO[k] = gen_facts(_src(nm)["lines"])
                    facts += [(pos[(nm, li)], f) for li, f in _MEMO[k] if (nm, li) in pos]
            else:
                facts = gen_facts(lines)
            if arg == "both":
                keys = [list(ks) for ks in skeys]
            else:
                keys = [[] for _ in lines]
            for li, f in facts:
                if 0 <= li < len(lines):
                    keys[li].append(f)
            keys = [k if k else ["(empty)"] for k in keys]
            out.append([top(key_scores(q[None], keys)) for q in Q])
        elif base == "amem":
            notes = LG(gen_amem, L, lines)
            if arg == "max":
                keys = [ks + ([n] if n else []) for ks, n in zip(skeys, notes)]
            else:
                keys = [[l + " " + (n or "")] for l, n in zip(lines, notes)]
            out.append([top(key_scores(q[None], keys)) for q in Q])
        elif base == "llmrank":
            n = int(arg or 50)
            cands = [top(s, n) for s in plain_s]
            picks = gen_llmrank([(q, [lines[i] for i in c]) for q, c in zip(qs, cands)])
            res = []
            for c, p in zip(cands, picks):
                pick = list(dict.fromkeys(c[i] for i in p))[:K]
                res.append((pick + [i for i in c if i not in pick])[:K])
            out.append(res)
        elif base == "regen":
            first = [top(s) for s in plain_s]
            drafts = gen_regen([(q, [lines[i] for i in f]) for q, f in zip(qs, first)])
            res = []
            for q, qv, d in zip(qs, Q, drafts):
                dv = embed([d], query=True)[0]
                vs = dv[None] if arg == "draft" else np.stack([qv, dv])
                res.append(top(key_scores(vs, skeys, "mean")))
            out.append(res)
        elif base == "d1":
            # arg "R" or "R.rank": R slots from the category pool; rank = plain | hyde | fact | ovl | ce
            r, _, how = arg.partition(".")
            how, _, gran = how.partition(".")
            how, _, wt = how.partition("@")
            wt = float(wt or 1)
            r = int(r or 3)
            lc = LG(gen_cats_lines, L, lines)
            qc = QG(gen_cats_reqs, L, qs)
            lcs = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in lc]
            G = {"g9": G9, "g5": G5}.get(gran)
            if G:
                lcs = [coarsen(c, G) for c in lcs]
            rng = np.random.default_rng(int(__import__("hashlib").md5(L["log"].encode()).hexdigest()[:8], 16))
            if gran == "shuf":  # control: same tags, moved to random lines
                lcs = [lcs[i] for i in rng.permutation(len(lcs))]
            if gran == "qrand":  # control: request tags replaced by random categories of the same count
                qc = [list(rng.choice(len(CATS), max(1, len(c or [])), replace=False)) for c in qc]
            df = np.bincount([c for cs in lcs for c in cs], minlength=len(CATS)) + 1
            if how in ("hyde", "ovlh"):
                notes = QG(gen_hyde, L, qs)
                rs = []
                for qv, nt, s0 in zip(Q, notes, plain_s):
                    if not nt:
                        rs.append(s0)
                        continue
                    hv = embed([nt])[0]
                    v = (hv + qv) / np.linalg.norm(hv + qv)
                    rs.append(key_scores(v[None], skeys))
            elif how == "fact":
                keys = [list(ks) for ks in skeys]
                for li, f in gen_facts(lines):
                    if 0 <= li < len(lines):
                        keys[li].append(f)
                rs = [key_scores(q[None], keys) for q in Q]
            else:
                rs = plain_s
            res = []
            for q, s, s0, c in zip(qs, rs, plain_s, qc):
                c = set(int(x) for x in (c or []) if str(x).lstrip("-").isdigit())
                if G:
                    c = coarsen(c, G)
                pool = [i for i, lcat in enumerate(lcs) if c & lcat]
                if how in ("ovl", "ovlh"):
                    key = lambda i: -(wt * sum(1.0 / np.log(df[x] + 1) for x in c & lcs[i]) + s[i])
                elif how == "ce" and pool:
                    cs = cross([(q, lines[i]) for i in pool])
                    cd = dict(zip(pool, cs))
                    key = lambda i: -cd[i]
                else:
                    key = lambda i: -s[i]
                pick = sorted(pool, key=key)[:r]
                rest = [i for i in top(s0, K + r) if i not in pick]
                res.append((pick + rest)[:K])
            out.append(res)
        elif base == "d3":
            # arg: miss (add the early request as a key to the needed line when the early search missed it),
            #      hit (ERM-style: add it only when the early search found the line), all (always), none
            early = gen_early(tasks)
            keys = [list(ks) for ks in skeys]
            E = embed([e or "" for e in early], query=True)
            for t, e, ev in zip(tasks, early, E):
                if not e:
                    continue
                got = any(i in top(key_scores(ev[None], skeys)) for i in t["target"])
                if arg == "all" or (arg == "miss" and not got) or (arg == "hit" and got):
                    for i in t["target"]:
                        keys[i].append(e)
            out.append([top(key_scores(q[None], keys)) for q in Q])
        elif base == "d4":
            # typed store: R slots from lines typed "standing", ranked by plain score
            r = int(arg or 6)
            ty = gen_type(lines)
            res = []
            for s in plain_s:
                pool = [i for i, x in enumerate(ty) if x == "standing"]
                pick = sorted(pool, key=lambda i: -s[i])[:r]
                rest = [i for i in top(s, K + r) if i not in pick]
                res.append((pick + rest)[:K])
            out.append(res)
        elif base == "d2":
            items = gen_drafts(qs)
            res = []
            for q, qv, it, s in zip(qs, Q, items, plain_s):
                it = [x for x in (it or []) if isinstance(x, str)]
                if not it:
                    res.append(top(s))
                    continue
                iv = embed(it, query=True)
                if arg.startswith("w"):  # weighted fusion: plain score + w * best item score
                    res.append(top(s + float(arg[1:]) * key_scores(iv, skeys, "max")))
                    continue
                vs = iv if arg == "items" else np.vstack([qv[None], iv])
                res.append(top(key_scores(vs, skeys, "max")))
            out.append(res)
        else:
            raise ValueError(name)
    return out
