"""Regenerate Figs. 2-4 from the leak-free three-way run (report quarter)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
SEEDS = [42, 7, 123]
NCLS, NC = 6, 400
THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
BS = [5, 10, 15, 20, 25]

d = json.load(open(f"{BASE}/threeway_cas.json"))
perm = np.random.RandomState(0).permutation(80868)
h, q = 80868 // 2, 80868 // 4
rep = perm[h + q:]
yte = np.load(f"{BASE}/booleanized_data.npz")["yte"][rep]

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "lines.linewidth": 1.4, "errorbar.capsize": 2})

# ---- pruning curve with accuracy, report quarter ----
ks = d["ks"]
f1c, accc = [], []
for s in SEEDS:
    Z = np.asarray(np.load(f"{BASE}/Zte_seed{s}.npy", mmap_mode="r")[rep], dtype=np.float32)
    W = np.load(f"{BASE}/W_seed{s}.npy")
    aw = np.abs(W)
    order = {c: c * NC + np.argsort(aw[c * NC:(c + 1) * NC], kind="stable") for c in range(NCLS)}
    f1s, accs = [], []
    for k in ks:
        Wm = W.copy()
        if k:
            Wm[np.concatenate([order[c][:k] for c in range(NCLS)])] = 0.0
        cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ Wm[c * NC:(c + 1) * NC]
                              for c in range(NCLS)])
        pred = np.argmax(cs, 1)
        f1s.append(f1_score(yte, pred, average="macro"))
        accs.append((pred == yte).mean())
    f1c.append(f1s); accc.append(accs)
f1c, accc = np.array(f1c), np.array(accc)
removed = np.array(ks) / NC * 100

fig, ax = plt.subplots(figsize=(3.2, 2.1))
for curve, lab, c in ((f1c, "macro-F1", "C0"), (accc, "accuracy", "C1")):
    m, sd = curve.mean(0), curve.std(0)
    ax.plot(removed, m, "-o", ms=2.5, color=c, label=lab)
    ax.fill_between(removed, m - sd, m + sd, color=c, alpha=0.2)
ax.axvline(84, color="k", ls="--", lw=0.8)
ax.text(84.5, ax.get_ylim()[0] + 0.02, "$k^\\star$=335 (84\\%)", fontsize=7, rotation=90, va="bottom")
ax.set_xlabel("clauses removed per class (%)")
ax.set_ylabel("score (report quarter)")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIGS}/clause_pruning.png", dpi=200); plt.close(fig)

# ---- pi_thr / B sweeps, full pool, report quarter ----
g = d["pools"]["full"]["grid"]
base_f1, base_acc = 0.6922, None
# base acc on rep: recompute once
baccs = []
for s in SEEDS:
    Z = np.asarray(np.load(f"{BASE}/Zte_seed{s}.npy", mmap_mode="r")[rep], dtype=np.float32)
    W = np.load(f"{BASE}/W_seed{s}.npy")
    cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ W[c * NC:(c + 1) * NC] for c in range(NCLS)])
    baccs.append((np.argmax(cs, 1) == yte).mean())
base_acc = float(np.mean(baccs))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.2))
xs = THRS
f1m = [g[f"15|{t}"]["rep_f1"] for t in xs]; f1s = [g[f"15|{t}"]["rep_f1_sd"] for t in xs]
acm = [g[f"15|{t}"]["rep_acc"] for t in xs]; acs = [g[f"15|{t}"]["rep_acc_sd"] for t in xs]
a1.errorbar(xs, f1m, yerr=f1s, marker="o", ms=3, label="macro-F1", color="C0")
a1.errorbar(xs, acm, yerr=acs, marker="s", ms=3, label="accuracy", color="C1")
a1.axhline(base_f1, color="C0", ls=":", lw=1); a1.axhline(base_acc, color="C1", ls=":", lw=1)
a1.set_xlabel("$\\pi_{\\mathrm{thr}}$"); a1.set_ylabel("score (report quarter)")
a1.set_title("(a) stability threshold, $B{=}15$", fontsize=8)
a1.legend(frameon=False, fontsize=7, loc="lower right")

bm = [g[f"{b}|0.8"]["rep_f1"] for b in BS]; bs = [g[f"{b}|0.8"]["rep_f1_sd"] for b in BS]
bam = [g[f"{b}|0.8"]["rep_acc"] for b in BS]; bas = [g[f"{b}|0.8"]["rep_acc_sd"] for b in BS]
a2.errorbar(BS, bm, yerr=bs, marker="o", ms=3, label="macro-F1", color="C0")
a2.errorbar(BS, bam, yerr=bas, marker="s", ms=3, label="accuracy", color="C1")
a2.axhline(base_f1, color="C0", ls=":", lw=1); a2.axhline(base_acc, color="C1", ls=":", lw=1)
a2.set_xlabel("complementary pairs $B$"); a2.set_title("(b) number of pairs, $\\pi_{\\mathrm{thr}}{=}0.8$", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_sweep.png", dpi=200); plt.close(fig)

# ---- pruned vs full pi_thr sweep (B=15 both pools) ----
fig, ax = plt.subplots(figsize=(3.2, 2.1))
for pool, lab, c in (("full", "full pool (2,400)", "C0"), ("pruned", "pruned pool (390)", "C2")):
    gg = d["pools"][pool]["grid"]
    m = [gg[f"15|{t}"]["rep_f1"] for t in THRS]; sds = [gg[f"15|{t}"]["rep_f1_sd"] for t in THRS]
    ax.errorbar(THRS, m, yerr=sds, marker="o", ms=3, label=lab, color=c)
ax.axhline(base_f1, color="C0", ls=":", lw=1, label="base TM (full)")
bp = d["pools"]["pruned"]["base_rep_f1"]
ax.axhline(bp, color="C2", ls=":", lw=1, label="base TM (pruned)")
ax.set_xlabel("$\\pi_{\\mathrm{thr}}$"); ax.set_ylabel("macro-F1 (report quarter)")
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGS}/pruned_vs_full.png", dpi=200); plt.close(fig)
print("figures written")
