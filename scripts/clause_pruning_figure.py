"""Figure: clause-weight pruning curve (3-seed mean +/- sd)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
D = json.load(open(f"{BASE}/clause_weight_pruning.json"))
ks = np.array(D["removed_per_class"])
F = np.array([r["macro_f1"] for r in D["runs"]])
A = np.array([r["accuracy"] for r in D["runs"]])
NC = D["clauses_per_class"]

fm, fs = F.mean(0), F.std(0)
am, asd = A.mean(0), A.std(0)
full = fm[0]

# knee: largest k still within 1 pp of the full-pool macro-F1
ok = np.where(fm >= full - 0.01)[0]
kk = ks[ok.max()]

fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.fill_between(ks, fm - fs, fm + fs, color="C1", alpha=.18)
ax.plot(ks, fm, "-", color="C1", lw=1.6, label="macro-F1")
ax.fill_between(ks, am - asd, am + asd, color="C0", alpha=.18)
ax.plot(ks, am, "-", color="C0", lw=1.6, label="accuracy")
ax.axhline(full, ls="--", lw=1, color="0.45", label="full pool (400/class)")
ax.axvline(kk, ls=":", lw=1.2, color="0.25")
ax.annotate(f"{kk}/{NC} removed\n({kk/NC*100:.0f}% of pool)\nwithin 1 pp",
            xy=(kk, fm[ok.max()]), xytext=(kk - 165, full - 0.075),
            fontsize=6.5, ha="left",
            arrowprops=dict(arrowstyle="->", lw=.8, color="0.25"))
ax.set_xlabel("clauses removed per class (lowest $|$weight$|$ first)")
ax.set_ylabel("score (3-seed mean)")
ax.set_xlim(0, NC)
ax.grid(ls="--", alpha=.3)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(f"{FIGS}/clause_pruning.png", dpi=200)
plt.close(fig)
print("saved figs/clause_pruning.png")
print(f"full macro-F1 {full:.4f}; within 1pp up to {kk} removed/class "
      f"({NC-kk} kept, {kk/NC*100:.0f}% pruned), macro-F1 {fm[ok.max()]:.4f}")
