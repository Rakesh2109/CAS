"""
All figures + LaTeX tables for the full-dataset CAS results.

  figs/full_pithr_acc_f1.png     1. pi_thr vs accuracy AND macro-F1 (3-seed)
  figs/full_B_acc.png            2. B (CPSS subsamples) vs accuracy / macro-F1
  figs/full_gridsearch.png       3. TM grid search: macro-F1 vs clauses / T / s
  figs/full_kappa_acc.png        4. kappa vs macro-F1 (per pi_thr)
  stdout: kappa x pi_thr table, binary F1/R/P/AUROC table, Table 3 (compaction)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
PI = [0.5, 0.6, 0.7, 0.8, 0.9]
BSN = [5, 10, 15, 20, 25]


def _agg(runs, path):
    """mean/std over seeds for a nested numeric field; path is list of keys."""
    vals = []
    for r in runs:
        x = r
        for k in path:
            x = x[k]
        vals.append(x)
    return float(np.mean(vals)), float(np.std(vals))


import sys
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "dedup"
def load(arm):
    return json.load(open(f"{BASE}/{PREFIX}_cas_{arm}.json"))


def fig_pithr(mc):
    runs = mc["runs"]
    accs = [[_agg(runs, ["B_sweep", "15", str(t), m])[0] for t in PI]
            for m in ("accuracy", "macro_f1")]
    accs_sd = [[_agg(runs, ["B_sweep", "15", str(t), m])[1] for t in PI]
               for m in ("accuracy", "macro_f1")]
    base_f1 = _agg(runs, ["base", "macro_f1"])
    base_acc = _agg(runs, ["base", "accuracy"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(PI, accs[0], yerr=accs_sd[0], fmt="s-", capsize=3, label="CAS accuracy")
    ax.errorbar(PI, accs[1], yerr=accs_sd[1], fmt="o-", capsize=3, label="CAS macro-F1")
    ax.axhline(base_acc[0], ls=":", c="C0", lw=1, label="base TM accuracy")
    ax.axhline(base_f1[0], ls="--", c="C1", lw=1, label="base TM macro-F1")
    ax.set_xlabel(r"$\pi_{\mathrm{thr}}$"); ax.set_ylabel("score (3-seed mean)")
    ax.set_xticks(PI); ax.grid(ls="--", alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIGS}/{PREFIX}_pithr_acc_f1.png", dpi=200)
    plt.close(fig)


def fig_B(mc):
    runs = mc["runs"]
    thr = "0.8"
    f1 = [_agg(runs, ["B_sweep", str(b), thr, "macro_f1"])[0] for b in BSN]
    f1s = [_agg(runs, ["B_sweep", str(b), thr, "macro_f1"])[1] for b in BSN]
    acc = [_agg(runs, ["B_sweep", str(b), thr, "accuracy"])[0] for b in BSN]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.errorbar(BSN, f1, yerr=f1s, fmt="o-", capsize=3, label=r"macro-F1 ($\pi_{\mathrm{thr}}{=}0.8$)")
    ax.plot(BSN, acc, "s--", label="accuracy")
    ax.set_xlabel("B (CPSS complementary pairs)")
    ax.set_ylabel("score (3-seed mean)")
    ax.set_xticks(BSN); ax.grid(ls="--", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/{PREFIX}_B_acc.png", dpi=200)
    plt.close(fig)


def fig_gridsearch():
    import matplotlib.ticker as mtick
    try:
        g = json.load(open(f"{BASE}/gridsearch_tm22.json"))["results"]
    except FileNotFoundError:
        print("no gridsearch_tm22.json - skip grid fig"); return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, key in zip(axes, ("clauses", "T", "s")):
        xs = sorted(set(r[key] for r in g))
        best = [max(r["best_macro_f1"] for r in g if r[key] == x) for x in xs]
        ax.plot(range(len(xs)), best, "o-")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(int(x)) if key != "s" else str(x) for x in xs],
                           rotation=45, fontsize=7)
        ax.set_xlabel(key); ax.grid(ls="--", alpha=.3)
    axes[0].set_ylabel("best macro-F1")
    axes[1].set_title("TM hyperparameter grid (22-feature, best-per-value)")
    fig.tight_layout(); fig.savefig(f"{FIGS}/{PREFIX}_gridsearch.png", dpi=200)
    plt.close(fig)


def fig_kappa(mc):
    runs = mc["runs"]
    ks = list(runs[0]["kappa_grid"].keys())
    kf = [float(k) for k in ks]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for thr in ("0.6", "0.7", "0.8"):
        m = [_agg(runs, ["kappa_grid", k, thr])[0] for k in ks]
        s = [_agg(runs, ["kappa_grid", k, thr])[1] for k in ks]
        ax.errorbar(kf, m, yerr=s, fmt="o-", capsize=3, label=fr"$\pi_{{\mathrm{{thr}}}}={thr}$")
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("CAS macro-F1 (3-seed mean)")
    ax.grid(ls="--", alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/{PREFIX}_kappa_acc.png", dpi=200)
    plt.close(fig)


def tables(mc, bn):
    runs = mc["runs"]
    ks = list(runs[0]["kappa_grid"].keys())
    print("\n=== kappa x pi_thr (3-seed mean CAS macro-F1, multiclass) ===")
    print(r"$\kappa\backslash\pi_{\mathrm{thr}}$ & " + " & ".join(PI := ["0.5","0.6","0.7","0.8","0.9"]) + r" \\")
    for k in ks:
        row = " & ".join(f"{_agg(runs, ['kappa_grid', k, t])[0]:.3f}" for t in PI)
        print(f"{float(k):.3g} & {row} \\\\")
    ur = " & ".join(f"{_agg(runs, ['B_sweep','15',t,'macro_f1'])[0]:.3f}" for t in PI)
    print(f"union grid & {ur} \\\\")

    print("\n=== Table 3: signature compaction (seed 42, pi_thr=0.8, B=15) ===")
    c = [r for r in runs if r["seed"] == 42][0]["compaction"]
    print(r"Family & Original & Certified $|S|$ & Final $|S^+|$ & $E[V]$ \\")
    for f, v in c.items():
        print(f"{f} & {v['orig']} & {v['S']} & {v['S_pos']} & $\\le${v['E_V']:.0f} \\\\")

    if bn is None:
        return
    print("\n=== Binary: F1 / R / P / AUROC (3-seed mean) ===")
    br = bn["runs"]
    def m(p): return _agg(br, p)
    print(r"Method & attack F1 & R & P & AUROC & AUPRC \\")
    print(f"Binary TM argmax & {m(['base','attack_f1'])[0]:.3f} & "
          f"{m(['base','attack_recall'])[0]:.3f} & {m(['base','attack_precision'])[0]:.3f} & "
          f"{m(['base','auroc'])[0]:.3f} & {m(['base','auprc'])[0]:.3f} \\\\")
    print(f"CAS signature & {m(['cas_thr0.8_B15','attack_f1'])[0]:.3f} & "
          f"{m(['cas_thr0.8_B15','attack_recall'])[0]:.3f} & {m(['cas_thr0.8_B15','attack_precision'])[0]:.3f} & "
          f"{m(['cas_thr0.8_B15','auroc'])[0]:.3f} & {m(['cas_thr0.8_B15','auprc'])[0]:.3f} \\\\")


def main():
    mc = load("multiclass")
    try:
        bn = load("binary")
    except FileNotFoundError:
        bn = None
    fig_pithr(mc); fig_B(mc); fig_kappa(mc)
    print(f"saved {PREFIX}_ figures")
    tables(mc, bn)


if __name__ == "__main__":
    main()
