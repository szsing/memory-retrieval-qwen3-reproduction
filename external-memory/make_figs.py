"""Figures for PAPER.html from results.jsonl (no model calls).  python make_figs.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {}
for l in open("results.jsonl"):
    r = json.loads(l)
    if r["kind"] == "ret" and r["split"] == "test":
        R[(r["slice"], r["method"])] = r["recall6"]
SL = {120: "len120", 240: "prefeval", 480: "len480", 960: "len960"}
C = {"d1": "#2a78d6", "plain": "#eb6834", "hyde": "#1baf7a", "d2qhyde": "#eda100", "g5": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def g(L, k, m):
    return R.get((SL[L] if k == 6 else f"{SL[L]}@k{k}", m))


def arms(k):
    return [("Tagged by activity (ours)", f"d1:{k}.ovl", C["d1"]), ("Plain search", "plain", C["plain"]),
            ("HyDE", "hyde:mix", C["hyde"]), ("doc2query + HyDE (equal calls)", "d2qhyde:3", C["d2qhyde"]),
            ("5 life scenes (CreaMem-style)", f"d1:{k}.plain.g5", C["g5"])]


# Figure 2: recall vs log length at K=6, and gain vs share of the log read
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
Ls = [120, 240, 480, 960]
for name, m, c in arms(6):
    ax[0].plot(Ls, [g(L, 6, m) for L in Ls], marker="o", ms=5, lw=2, color=c, label=name)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(Ls); ax[0].set_xticklabels(Ls)
ax[0].set_xlabel("Lines in the user's chat history"); ax[0].set_ylabel("Earlier fact found in the 6 notes read")
ax[0].set_ylim(0, 0.85); ax[0].grid(axis="y", color=GRID); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("A. Six notes read", fontsize=10, color=INK, loc="left")
pts = []
for k in [1, 2, 3, 6, 10, 12, 20, 24]:
    for L in Ls:
        d = g(L, k, f"d1:{k}.ovl")
        base = [g(L, k, m) for m in ["plain", "hyde:mix", "d2q:3", "amem:max", "mem0:both"]]
        base = [b for b in base if b is not None]
        if d is not None and base:
            pts.append((k / L, d - max(base), L))
mk = {120: "o", 240: "s", 480: "^", 960: "D"}
for L in Ls:
    p = [(x, y) for x, y, l in pts if l == L]
    ax[1].scatter([x for x, _ in p], [y for _, y in p], marker=mk[L], s=36, color=C["d1"], edgecolor="white",
                  linewidth=1, label=f"{L} lines")
ax[1].axhline(0, color=MUTED, lw=1); ax[1].axhline(0.10, color=MUTED, lw=1, ls="--")
ax[1].text(0.0032, 0.112, "+0.10 bar", fontsize=8, color=MUTED)
ax[1].set_xscale("log"); ax[1].set_xlabel("Share of the history read (notes read / lines)")
ax[1].set_ylabel("Gain over the best baseline"); ax[1].grid(axis="y", color=GRID)
ax[1].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].set_title("B. Every grid cell", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig("fig_scale.png", dpi=170); plt.close(fig)

# Figure 3: what the gain needs (vocabulary size, rarity weighting, right tags)
rows = [("50 activities, rarity-weighted", "d1:6.ovl"), ("50 activities, pool by plain score", "d1:6"),
        ("9 groups, rarity-weighted", "d1:6.ovl.g9"), ("5 groups, rarity-weighted", "d1:6.ovl.g5"),
        ("5 groups, pool by plain score", "d1:6.plain.g5"), ("Tags shuffled across facts", "d1:6.ovl.shuf"),
        ("Random request tags", "d1:6.ovl.qrand")]
fig, ax = plt.subplots(figsize=(7.4, 3.2))
v = [R[("prefeval", m)] for _, m in rows]
ax.barh(range(len(rows)), v, height=0.6, color=[C["d1"]] + ["#9cbfe9"] * 4 + ["#c9c9c5"] * 2)
for i, x in enumerate(v):
    ax.text(max(x, 0.43 if 0.36 < x < 0.43 else x) + 0.01, i, f"{x:.2f}", va="center", fontsize=9, color=INK)
ax.axvline(R[("prefeval", "plain")], color=C["plain"], lw=1.5, ls="--")
ax.text(R[("prefeval", "plain")] + 0.005, len(rows) - 0.4, "plain search 0.42", fontsize=8, color=MUTED)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows]); ax.invert_yaxis()
ax.set_xlim(0, 0.75); ax.set_xlabel("Earlier fact found in the 6 notes read")
fig.tight_layout(); fig.savefig("fig_controls.png", dpi=170); plt.close(fig)

# Figure 4: applied rate follows recall
fig, ax = plt.subplots(figsize=(5, 3.4))
pts = [("Plain", 0.472, 0.433, C["plain"]), ("HyDE", 0.544, 0.400, C["hyde"]),
       ("Ours", 0.633, 0.533, C["d1"]), ("Fact handed over", 1.0, 0.656, "#6b6b68")]
xs = [0, 1]; ax.plot(xs, [0.17 + 0.48 * x for x in xs], color=GRID, lw=2, zorder=0)
for n, x, y, c in pts:
    ax.scatter([x], [y], s=60, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(n, (x, y), xytext=(6, -12 if n == "HyDE" else 6), textcoords="offset points", fontsize=9, color=INK)
ax.set_xlim(0, 1.1); ax.set_ylim(0, 0.8); ax.grid(axis="y", color=GRID)
ax.set_xlabel("Earlier fact found in the notes (answered requests)"); ax.set_ylabel("Answer applies the fact")
fig.tight_layout(); fig.savefig("fig_answer.png", dpi=170); plt.close(fig)
print("ok")
