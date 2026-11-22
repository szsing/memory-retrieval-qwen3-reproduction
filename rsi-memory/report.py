"""Print per-design round scores from results.jsonl and draw fig_rounds.png (unseen-domain recall per round)."""
import json, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = collections.defaultdict(dict)
for l in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(l)
    if r.get("kind") == "round":
        R[(r["bed"], r["design"])][r["round"]] = r   # last run of a design wins
for (bed, d), rs in sorted(R.items()):
    ks = sorted(rs)
    f = lambda k: "/".join(str(rs[i].get(k)) for i in ks)
    if bed == "inmind":
        print(f"{d:34s} A_off {f('testA_off')}  B_off {f('testB_off')}  B_dense {f('testB_dense')}  naive {f('naive_off')}")
    else:
        print(f"{bed}:{d:25s} test_dense {f('testB_dense')}")
show = ["d1+tags+impkeys+practice", "d1+tags+practice", "tags+practice:r8", "tags+practice", "impkeys+practice", "exp+rulewriter", "d1+freetags", "d1", "d2q", "exp", "plain"]
plt.figure(figsize=(7, 4.2))
for d in show:
    rs = R.get(("inmind", d))
    if not rs:
        continue
    ks = sorted(rs)
    ys = [rs[k]["testB_off"] for k in ks]
    if len(ks) == 1:
        ks, ys = [0, 8], ys * 2
    plt.plot(ks, ys, marker="o" if len(set(ys)) > 1 else None, ls="-" if len(set(ys)) > 1 else "--", label=d)
plt.xlabel("learning round"); plt.ylabel("recall@6, unseen domains (61 tasks)")
plt.title("InMind official log (241 lines): learned on 5 domains, scored on the other 5")
plt.xticks(range(9)); plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_rounds.png"), dpi=130)
