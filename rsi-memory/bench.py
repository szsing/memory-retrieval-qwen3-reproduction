"""Benchmark harness with overfit checks.

  python bench.py <design> [--held test]        design parts joined by +: plain, d1, prop (more can be registered)

Why: a design search maximises one score, so it can fit that benchmark instead of the problem. The harness keeps
the search bed and the check beds apart and prints both, so a gain that appears only on the search bed is visible.

  SEARCH bed (tune and read failures here):
    inmind      InMind 2607.24368, 61 unseen-domain tasks. official = fact among 240 background lines; dense = all
                125 facts + background (365 lines). Hyperparameters may only be picked on InMind learning-domain tasks.
  CHECK beds (aggregate numbers only; never read per-task failures, never tune on them):
    harm        direct-question recall on InMind (the request names its need). A design must not break plain search.
    prefeval    PrefEval, 45 test requests over a 478-line store (numbers already seen in round 1 and 2).
    locomo      LoCoMo, 89 ordinary memory questions over 10 conversations of 240 lines (normal memory, not
                hidden constraints): checks that a design does not break everyday recall.
    lucid-val   LUCid 2604.26996 (CC BY 4.0), a benchmark this project never used before 2026-11-17: latent user
                context (age, country, religion, health, domain, style) in semantically distant sessions. Each
                request searches its own user's history (its latent sessions + 2 unrelated sessions). Used for checks.
    lucid-test  disjoint LUCid requests, scored only for frozen designs (--held test); each look is logged.
    big         scale: InMind dense plus ~2k unrelated user lines from LUCid sessions (no answers), 61 tasks.
Overfit signal: gain over the reference design on SEARCH minus gain on CHECK beds. Every run appends one row to
bench_runs.jsonl (aggregates only).
Model calls: Claude Haiku through lm.py (cached). Embeddings: BGE-small.
"""
import argparse
import json
import os
import random
import time

import numpy as np

import bridge
import rsi
from rsi import Store, K, h

HERE = os.path.dirname(os.path.abspath(__file__))
LUCID = os.path.join(HERE, "bench_data", "lucid", "lucid.json")


# ---------------- beds: dict(lines, tasks); task = dict(q, targets, pool=(lo, hi) or None) ----------------

def bed_inmind_parts():
    b = rsi.bed_inmind()
    nb = b["nb"]
    tB = [dict(q=t["q"], targets=[t["target"]], pool=None, off=True) for t in b["testB"]]
    harm = [dict(q=t["naive"], targets=[t["target"]], pool=None, off=True) for t in b["testA"] + b["testB"]]
    return b["lines"], nb, tB, harm


def bed_prefeval():
    b = rsi.bed_prefeval()
    return dict(lines=b["lines"], tasks=[dict(q=t["q"], targets=[t["target"]], pool=None) for t in b["testB"]])


def bed_locomo(which="check"):
    """LoCoMo chats 0-4 = tuning (ordinary questions the search may see), chats 5-9 = check."""
    L = [json.loads(l) for l in open(os.path.join(HERE, "..", "external-memory", "bed", "locomo.jsonl"))]
    L = L[:5] if which == "dev" else L[5:]
    lines, tasks = [], []
    for g in L:
        lo = len(lines)
        lines += g["lines"]
        for t in g["tasks"]:
            tasks.append(dict(q=t["q"], targets=[lo + i for i in t["target"]], pool=(lo, len(lines))))
    return dict(lines=lines, tasks=tasks)


def bed_prefeval_logs(idx):
    """PrefEval logs searched one at a time (each log is its own store of ~240 lines, 45 requests). Logs 0-1 tune,
    logs 6-13 check (logs 4-5 are the older 45-request check bed)."""
    L = [json.loads(l) for l in open(os.path.join(HERE, "..", "external-memory", "bed", "prefeval.jsonl"))]
    lines, tasks = [], []
    for i in idx:
        g = L[i]
        lo = len(lines)
        lines += g["lines"]
        for t in g["tasks"]:
            tasks.append(dict(q=t["q"], targets=[lo + x for x in t["target"]], pool=(lo, len(lines))))
    return dict(lines=lines, tasks=tasks)


def _user_turns(sess):
    return [(t["content"].replace("\n", " ").strip(), bool(t.get("has_answer"))) for t in sess if t["role"] == "user"]


def lucid_split():
    """Fixed split of LUCid requests by hash of query_id: val = 120, test = 240, the rest only supplies unrelated
    sessions. Never re-drawn."""
    D = json.load(open(LUCID))
    D = [d for d in D if any(t.get("has_answer") for s in d["latent_sessions"] for t in s if t["role"] == "user")]
    D.sort(key=lambda d: h(d["query_id"]))
    return D[:120], D[120:360], D[360:]


def lucid_other(rest, n_sess, seed, avoid=()):
    """Unrelated sessions taken from LUCid requests outside val and test."""
    pool = [(sid, s) for d in rest for sid, s in zip(d["answer_session_ids"], d["latent_sessions"]) if sid not in avoid]
    rng = random.Random(seed)
    return [pool[i][1] for i in rng.sample(range(len(pool)), min(n_sess, len(pool)))]


def bed_lucid(which, n_shared=100):
    """Each request searches its own user's history: its latent sessions (LUCid marks the answer turns) plus 100
    unrelated sessions shared by every request of the split (other LUCid users, so they hold competing facts).
    Dimensions: health, country, age, domain, style, religion."""
    val, test, rest = lucid_split()
    items = val if which == "val" else test
    lines, tasks, own = [], [], []
    for d in items:
        lo = len(lines)
        tg = []
        for s in d["latent_sessions"]:
            for txt, ans in _user_turns(s):
                if ans:
                    tg.append(len(lines))
                lines.append(txt)
        own.append((lo, len(lines)))
        tasks.append(dict(q=d["query"], targets=tg, dim=d["query_dimension"]))
    s0 = len(lines)
    for s in lucid_other(rest, n_shared, seed=1 if which == "val" else 2):
        lines += [txt for txt, _ in _user_turns(s)]
    shared = np.arange(s0, len(lines))
    for t, (lo, hi) in zip(tasks, own):
        t["pool"] = np.concatenate([np.arange(lo, hi), shared])
    return dict(lines=lines, tasks=tasks)


def bed_big(n_extra=2000):
    lines, nb, tB, _ = bed_inmind_parts()
    _, _, rest = lucid_split()
    extra = []
    for s in lucid_other(rest, 1500 if n_extra <= 2000 else 10 ** 6, seed=3):
        extra += [txt for txt, ans in _user_turns(s) if not ans and len(txt) > 20]
    extra = extra[:n_extra]
    return dict(lines=lines + extra, tasks=[dict(q=t["q"], targets=t["targets"], pool=None) for t in tB])


IMPLEX = os.path.join(HERE, "bench_data", "implexconv", "implexconv_opposed_trim.json")
LME = os.path.join(HERE, "bench_data", "longmemeval", "oracle.json")


def bed_implex(which):
    """ImplexConv opposed (2503.07018, MIT). Each user's pool: the sessions that hold the hidden trait plus 30 other
    sessions of the same user, user turns only. A hit = any top-6 line from an evidence session (session-level labels).
    Frozen portion: val = dataset rows 0-29, test = rows 800-829 (fetched 2026-11-17, never used for design)."""
    users = json.load(open(IMPLEX))[which]
    lines, tasks = [], []
    for u in users:
        lo = len(lines); where = {}
        for sid, turns in u["sessions"].items():
            for t in turns:
                where.setdefault(sid, []).append(len(lines)); lines.append(t[:1000])
        for q in u["qa"]:
            tasks.append(dict(q=q["q"], targets=[i for sid in q["ev"] for i in where.get(sid, [])], pool=(lo, len(lines))))
    return dict(lines=lines, tasks=tasks)


def bed_lme(qtype, n_shared=60):
    """LongMemEval oracle (MIT). Each question's pool: its own evidence sessions (user turns) plus n_shared sessions of
    other questions shared by all (other users). A hit = a user turn marked has_answer in the top 6.
    single-session-preference = indirect requests (30); single-session-user = ordinary recall (70). No tuning on either."""
    d = json.load(open(LME))
    items = [x for x in d if x["question_type"] == qtype]
    rest = [x for x in d if x["question_type"] != qtype]
    lines, tasks, own = [], [], []
    for x in items:
        lo = len(lines); tg = []
        for sess in x["haystack_sessions"]:
            for t in sess:
                if t["role"] != "user":
                    continue
                if t.get("has_answer"):
                    tg.append(len(lines))
                lines.append(t["content"][:1000])
        own.append((lo, len(lines)))
        tasks.append(dict(q=x["question"], targets=tg))
    rng = random.Random(7)
    pool_s = [s for x in rest for s in x["haystack_sessions"]]
    s0 = len(lines)
    for sess in rng.sample(pool_s, n_shared):
        lines += [t["content"][:1000] for t in sess if t["role"] == "user"]
    shared = np.arange(s0, len(lines))
    for t, (lo, hi) in zip(tasks, own):
        t["pool"] = np.concatenate([np.arange(lo, hi), shared])
    return dict(lines=lines, tasks=[t for t in tasks if t["targets"]])


# ---------------- designs ----------------
W = {"prop": 2.0}
MIX = ["rrf", None]   # picked on the tuning set only (--dev); frozen for every check bed
CTRL = [""]


def _pool(p, n):
    return np.arange(n) if p is None else (np.arange(*p) if isinstance(p, tuple) else np.asarray(p))


def scores(lines, qs, parts, pools=None):
    """pools: each request's candidate lines (None = whole store); the cpin mix pins dense top-k inside the pool."""
    st = Store(dict(lines=lines, nb=None))
    q1 = None
    if "d1" in parts:
        import methods as Mx
        st.d1l = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in Mx.gen_cats_lines([l[:600] for l in lines])]
        q1 = [set(int(x) for x in (c or []) if str(x).isdigit()) for c in Mx.gen_cats_reqs(qs)]
    ph = None
    if "hyde" in parts:          # rival: the model writes a made-up earlier user note that would matter (HyDE-style)
        import methods as Mx
        ph = [[n] if n else [] for n in Mx.gen_hyde(qs)]
    S = st.scores(qs, ph, None, q1)
    S0 = st.scores(qs) if "d1" in parts else S   # plain dense cosines, for the gate
    bridge.QGATE[0] = MIX[0] in ("qgate", "qthr")
    if "prop" in parts:
        pidx = bridge.PropIndex(bridge.props_lines(lines))
        rp = bridge.props_reqs(qs)
        if CTRL[0] == "shuf":     # control: each request gets another request's properties
            k_ = len(rp) // 2; rp = rp[k_:] + rp[:k_]
        B = pidx.bonus(rp)
        if CTRL[0] == "null":     # control: a line with any property gets the same bonus (a "states a fact" prior)
            has = np.zeros(pidx.n); has[pidx.own] = 1.0
            B = [(has if r else 0.0) + 1e-3 * s0 for r, s0 in zip(rp, S)]   # ties broken by the dense score

        def rr(x):
            r = np.empty(len(x)); r[np.argsort(-x)] = np.arange(len(x)); return 1.0 / (60 + r)
        if MIX[0] == "rrf":
            S = [rr(s) + W["prop"] * rr(b) for s, b in zip(S, B)]
        elif MIX[0] in ("thr", "qthr"):   # only strong property matches count: cosine above theta
            th = MIX[1]
            S = [rr(s) + W["prop"] * np.where(b > th, rr(b), 0.0) for s, b in zip(S, B)]
        elif MIX[0] == "cgate":    # separate help-or-recall call; recall questions keep the dense ranking alone
            G = bridge.gate_reqs(qs)
            S = [rr(s) + (W["prop"] * rr(b) if g else 0.0) for s, b, g in zip(S, B, G)]
        elif MIX[0] == "cpin":     # gate + dense keeps its top k in place; the rest ranked by the weighted mix
            k = int(MIX[1]); G = bridge.gate_reqs(qs); out = []
            for i, (s, b, g) in enumerate(zip(S, B, G)):
                if not g:
                    out.append(s); continue
                f = rr(s) + W["prop"] * rr(b)
                ix = _pool(pools[i] if pools is not None else None, len(s))
                f[ix[np.argsort(-s[ix])[:k]]] += 100.0   # pin inside the request's own pool (fixed 2026-11-17)
                out.append(f)
            S = out
        elif MIX[0] == "cpthr":    # gate + pin dense top-1 in pool + only property matches above cosine theta count
            th = MIX[1]; G = bridge.gate_reqs(qs); out = []
            for i, (s, b, g) in enumerate(zip(S, B, G)):
                if not g:
                    out.append(s); continue
                f = rr(s) + W["prop"] * np.where(b > th, rr(b), 0.0)
                ix = _pool(pools[i] if pools is not None else None, len(s))
                f[ix[np.argsort(-s[ix])[:1]]] += 100.0
                out.append(f)
            S = out
        elif MIX[0] == "cpconf":   # gate + pin + store check: join only when dense's best in-pool score is below tau
            tau = MIX[1]; G = bridge.gate_reqs(qs); out = []
            for i, (s, b, g) in enumerate(zip(S, B, G)):
                ix = _pool(pools[i] if pools is not None else None, len(s))
                if not g or s[ix].max() >= tau:
                    out.append(s); continue
                f = rr(s) + W["prop"] * rr(b)
                f[ix[np.argsort(-s[ix])[:1]]] += 100.0
                out.append(f)
            S = out
        elif MIX[0] == "pick":     # gate + pin; Haiku picks up to 6 notes from the union of dense top 6 and join top 6
            G = bridge.gate_reqs(qs); out = []; todo = []
            for i, (s, b, g) in enumerate(zip(S, B, G)):
                ix = _pool(pools[i] if pools is not None else None, len(s))
                if not g:
                    out.append(s); continue
                f = rr(s) + W["prop"] * rr(b)
                f[ix[np.argsort(-s[ix])[:1]]] += 100.0
                c = list(dict.fromkeys([int(j) for j in ix[np.argsort(-s[ix])[:K]]] + [int(j) for j in ix[np.argsort(-f[ix])[:K]]]))
                if CTRL[0] == "wide":      # dense top 12 plus join top 6: Haiku sees both candidate sets in full
                    c = list(dict.fromkeys([int(j) for j in ix[np.argsort(-s[ix])[:2 * K]]] + [int(j) for j in ix[np.argsort(-f[ix])[:K]]]))
                if CTRL[0] == "nojoin":    # rival: Haiku re-ranks the dense top 12 only (RankGPT-style), no join candidates
                    c = [int(j) for j in ix[np.argsort(-s[ix])[:2 * K]]]
                out.append(f); todo.append((i, c))
            picks = bridge.pick_notes([qs[i] for i, _ in todo], [[lines[j] for j in c] for _, c in todo])
            for (i, c), pk in zip(todo, picks):
                f = out[i].copy()
                for r, n in enumerate(pk):
                    f[c[n]] += 1000.0 - r
                out[i] = f
            S = out
        elif MIX[0] == "qgate":    # request-side gate: no properties (a recall question) -> dense ranking alone
            S = [rr(s) + (W["prop"] * rr(b) if b.any() else 0.0) for s, b in zip(S, B)]
        elif MIX[0] in ("slot", "cslot"):   # dense keeps its top k lines, the property ranking fills the other 6 - k
            k = int(MIX[1])
            G = bridge.gate_reqs(qs) if MIX[0] == "cslot" else [True] * len(S)
            out = []
            for i, (s, b, g) in enumerate(zip(S, B, G)):
                if not g:
                    out.append(s); continue
                f = np.zeros(len(s))
                ix = _pool(pools[i] if pools is not None else None, len(s))
                top = list(ix[np.argsort(-s[ix])[:k]])
                for j in ix[np.argsort(-b[ix])]:
                    if len(top) >= K:
                        break
                    if j not in top:
                        top.append(j)
                f[top] = np.arange(len(top), 0, -1) + 1.0
                out.append(f + rr(s) * 1e-3)
            S = out
        elif MIX[0] == "gate":     # join only when dense search is unsure: top dense cosine below tau
            tau = MIX[1]
            S = [rr(s) + (W["prop"] if d.max() < tau else 0.0) * rr(b) for s, b, d in zip(S, B, S0)]
    return S


def recall(tasks, S, nb=None):
    out = []
    for t, s in zip(tasks, S):
        if t.get("off"):          # InMind official: rank the fact among the 240 background lines only
            tg = t["targets"][0]
            out.append(float((s[:nb] > s[tg]).sum() < K))
            continue
        p = t["pool"]
        if p is None:
            p = np.arange(len(s))
        elif isinstance(p, tuple):
            p = np.arange(*p)
        top = p[np.argsort(-s[p])[:K]]
        out.append(float(any(x in top for x in t["targets"])))
    return out


def dev(design):
    """InMind learning-domain tasks only (stream + gate, 40): indirect official + dense, and direct-question recall."""
    parts = set(design.split("+"))
    b = rsi.bed_inmind()
    T = b["stream"] + b["gate"]
    lines, nb = b["lines"], b["nb"]
    ind = [dict(q=t["q"], targets=[t["target"]], pool=None, off=True) for t in T]
    nav = [dict(q=t["naive"], targets=[t["target"]], pool=None, off=True) for t in T]
    S = scores(lines, [t["q"] for t in ind], parts)
    o = np.mean(recall(ind, S, nb)); d = np.mean(recall([dict(t, off=False) for t in ind], S))
    n = np.mean(recall(nav, scores(lines, [t["q"] for t in nav], parts), nb))
    lc = bed_locomo("dev")
    c = np.mean(recall(lc["tasks"], scores(lc["lines"], [t["q"] for t in lc["tasks"]], parts, [t["pool"] for t in lc["tasks"]])))
    pe = bed_prefeval_logs([0, 1])
    p = np.mean(recall(pe["tasks"], scores(pe["lines"], [t["q"] for t in pe["tasks"]], parts, [t["pool"] for t in pe["tasks"]])))
    return o, d, n, c, p


SIZE = dict(nshared=100, nextra=2000)


def run(design, held, beds=None):
    parts = set(design.split("+"))
    res = {}
    want = lambda n: beds is None or n in beds
    lines, nb, tB, harm = bed_inmind_parts()
    if want("inmind"):
        S = scores(lines, [t["q"] for t in tB], parts)
        res["inmind_off"] = recall(tB, S, nb)
        res["inmind_dense"] = recall([dict(t, off=False) for t in tB], S)
    if want("harm"):
        S = scores(lines, [t["q"] for t in harm], parts)
        res["harm"] = recall(harm, S, nb)
    for name, fn in (("prefeval", bed_prefeval), ("locomo", lambda: bed_locomo("check")), ("prefeval8", lambda: bed_prefeval_logs(range(6, 14))), ("lucid_" + held, lambda: bed_lucid(held, SIZE["nshared"])),
                     ("big", lambda: bed_big(SIZE["nextra"])), ("implex_" + held, lambda: bed_implex(held)),
                     ("lmepref", lambda: bed_lme("single-session-preference")), ("lmeuser", lambda: bed_lme("single-session-user"))):
        if not want(name.split("_")[0]):
            continue
        b = fn()
        res[name] = recall(b["tasks"], scores(b["lines"], [t["q"] for t in b["tasks"]], parts, [t["pool"] for t in b["tasks"]]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--held", default="val", choices=["val", "test"])
    ap.add_argument("--mix", default="rrf", help="rrf | qgate | slot:K | gate:TAU")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--ctrl", default="", help="null = every line with any property gets the same bonus; shuf = each request gets another request's properties")
    ap.add_argument("--beds", default="", help="comma list: inmind,harm,prefeval,locomo,prefeval8,lucid,big,implex,lmepref,lmeuser (default all)")
    ap.add_argument("--nshared", type=int, default=100, help="LUCid: unrelated sessions of other users in every pool")
    ap.add_argument("--nextra", type=int, default=2000, help="big: unrelated user lines added to InMind")
    ap.add_argument("--pv", type=int, default=1, help="property prompt version (bridge.py)")
    ap.add_argument("--dev", action="store_true", help="only print InMind learning-domain scores (for picking)")
    a = ap.parse_args()
    W["prop"] = a.w
    CTRL[0] = a.ctrl
    bridge.PV[0] = a.pv
    m = a.mix.split(":")
    MIX[0], MIX[1] = m[0], (float(m[1]) if len(m) > 1 else None)
    if a.dev:
        o, d, n, c, p = dev(a.design)
        print(f"[dev {a.design} {a.mix} w={W['prop']} pv={a.pv}] off {o:.3f} dense {d:.3f} direct {n:.3f} locomo {c:.3f} prefeval {p:.3f}  sum {o + d + n + c + p:.3f}")
        return
    c0 = rsi.lm.STATS["calls"] + rsi.lm.STATS["cached"]
    SIZE["nshared"], SIZE["nextra"] = a.nshared, a.nextra
    res = run(a.design, a.held, set(a.beds.split(",")) if a.beds else None)
    row = {k: [int(sum(v)), len(v)] for k, v in res.items()}
    print(f"[{a.design} {a.mix}] " + "  ".join(f"{k} {n}/{m}" for k, (n, m) in row.items()))
    with open(os.path.join(HERE, "bench_runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M"), design=a.design, mix=a.mix, w=a.w, held=a.held, ctrl=a.ctrl, pv=a.pv, nshared=a.nshared, nextra=a.nextra, scores=row,
                                calls=rsi.lm.STATS["calls"] + rsi.lm.STATS["cached"] - c0)) + "\n")


if __name__ == "__main__":
    main()
