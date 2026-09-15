"""Figures + LaTeX table for the ORIGINAL-protocol sensitivity sweeps.

  figs/old_pithr.png   pi_thr vs macro-F1 AND accuracy (3-seed mean +/- sd)
  figs/old_B.png       B vs macro-F1 AND accuracy
  figs/old_kappa.png   kappa vs macro-F1, per pi_thr
  stdout: kappa x pi_thr LaTeX table body
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
D = json.load(open(f"{BASE}/old_sweeps_final.json"))
R = D["runs"]
PI = [str(t) for t in D["pi_thrs"]]
BSN = D["B_snap"]
KS = list(R[0]["kappa_grid"].keys())


def agg(path):
    v = []
    for r in R:
        x = r
        for k in path:
            x = x[k]
        v.append(x)
    return float(np.mean(v)), float(np.std(v))


base_f1 = agg(["base_macro_f1"])
base_acc = agg(["base_accuracy"])
x = [float(t) for t in PI]

# --- pi_thr ---
f1m = [agg(["B_sweep", "15", t, "macro_f1"])[0] for t in PI]
f1s = [agg(["B_sweep", "15", t, "macro_f1"])[1] for t in PI]
acm = [agg(["B_sweep", "15", t, "accuracy"])[0] for t in PI]
acs = [agg(["B_sweep", "15", t, "accuracy"])[1] for t in PI]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.errorbar(x, f1m, yerr=f1s, fmt="o-", capsize=3, color="C1", label="CAS macro-F1")
ax.errorbar(x, acm, yerr=acs, fmt="s-", capsize=3, color="C0", label="CAS accuracy")
ax.axhline(base_f1[0], ls="--", lw=1, color="C1", label="base TM macro-F1")
ax.axhline(base_acc[0], ls=":", lw=1, color="C0", label="base TM accuracy")
ax.set_xlabel(r"$\pi_{\mathrm{thr}}$"); ax.set_ylabel("score (3-seed mean)")
ax.set_xticks(x); ax.grid(ls="--", alpha=.3); ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGS}/old_pithr.png", dpi=200); plt.close(fig)

# --- B ---
bf = [agg(["B_sweep", str(b), "0.8", "macro_f1"])[0] for b in BSN]
bfs = [agg(["B_sweep", str(b), "0.8", "macro_f1"])[1] for b in BSN]
ba = [agg(["B_sweep", str(b), "0.8", "accuracy"])[0] for b in BSN]
bas = [agg(["B_sweep", str(b), "0.8", "accuracy"])[1] for b in BSN]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.errorbar(BSN, bf, yerr=bfs, fmt="o-", capsize=3, color="C1", label="CAS macro-F1")
ax.errorbar(BSN, ba, yerr=bas, fmt="s-", capsize=3, color="C0", label="CAS accuracy")
ax.axhline(base_f1[0], ls="--", lw=1, color="C1", label="base TM macro-F1")
ax.set_xlabel(r"$B$ (CPSS complementary pairs)")
ax.set_ylabel(r"score at $\pi_{\mathrm{thr}}{=}0.8$")
ax.set_xticks(BSN); ax.grid(ls="--", alpha=.3); ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGS}/old_B.png", dpi=200); plt.close(fig)

# --- kappa ---
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for t in ("0.6", "0.8", "0.9"):
    m = [agg(["kappa_grid", k, t, "macro_f1"])[0] for k in KS]
    s = [agg(["kappa_grid", k, t, "macro_f1"])[1] for k in KS]
    ax.errorbar([float(k) for k in KS], m, yerr=s, fmt="o-", capsize=3,
                label=fr"$\pi_{{\mathrm{{thr}}}}={t}$")
ax.axhline(base_f1[0], ls="--", lw=1, color="0.4", label="base TM")
ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$")
ax.set_ylabel("CAS macro-F1 (3-seed mean)")
ax.grid(ls="--", alpha=.3, which="both"); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIGS}/old_kappa.png", dpi=200); plt.close(fig)

print("saved old_pithr.png old_B.png old_kappa.png\n")
print("=== kappa x pi_thr LaTeX rows (3-seed mean CAS macro-F1) ===")
for k in KS:
    kv = float(k)
    e = int(np.floor(np.log10(kv)))
    mant = kv / 10 ** e
    lab = fr"${mant:.1f}\!\times\!10^{{{e}}}$"
    row = " & ".join(f"{agg(['kappa_grid', k, t, 'macro_f1'])[0]:.3f}" for t in PI)
    print(f"{lab} & {row} \\\\")
print(r"\midrule")
ur = " & ".join(f"{agg(['B_sweep','15',t,'macro_f1'])[0]:.3f}" for t in PI)
print(f"union grid (method) & {ur} \\\\")
print()
print("base macro-F1 %.4f+/-%.4f  acc %.4f" % (base_f1[0], base_f1[1], base_acc[0]))
print("pi_thr macroF1:", [f"{v:.3f}" for v in f1m])
print("pi_thr acc    :", [f"{v:.3f}" for v in acm])
print("B macroF1     :", [f"{v:.3f}" for v in bf])
print("B acc         :", [f"{v:.3f}" for v in ba])
