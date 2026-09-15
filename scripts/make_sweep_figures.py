"""
Figures for the CAS hyperparameter-sensitivity results:
  * pi_thr_vs_accuracy.png   -- CAS macro-F1 vs pi_thr, 3-seed mean +/- sd
  * kappa_vs_accuracy.png     -- CAS macro-F1 vs kappa (CPSS inverse-reg), 3-seed
  * kappa_pi_grid.png         -- kappa x pi_thr macro-F1 heatmap (3-seed mean)
and a LaTeX fragment for the kappa x pi_thr grid table.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load():
    ms = json.load(open(f"{BASE}/iotm_multiseed_perclass_B15.json"))
    kg = json.load(open(f"{BASE}/kappa_grid_sweep.json"))
    return ms, kg


def fig_pi_thr_vs_accuracy(ms):
    base_m = ms["tm_argmax_macro_f1_mean"]
    base_s = ms["tm_argmax_macro_f1_std"]
    means = [ms[f"cas_thr{t}_macro_f1_mean"] for t in PI_THRS]
    stds = [ms[f"cas_thr{t}_macro_f1_std"] for t in PI_THRS]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.axhspan(base_m - base_s, base_m + base_s, color="0.85", zorder=0)
    ax.axhline(base_m, color="0.4", ls="--", lw=1, label="weighted-TM argmax (base)")
    ax.errorbar(PI_THRS, means, yerr=stds, fmt="o-", capsize=3, color="C0",
                label="CAS attribution")
    ax.set_xlabel(r"$\pi_{\mathrm{thr}}$ (stability threshold)")
    ax.set_ylabel("macro-F1 (3 seeds)")
    ax.set_xticks(PI_THRS)
    ax.grid(ls="--", alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/pi_thr_vs_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_kappa_vs_accuracy(kg, ms):
    kappas = kg["kappa_grid"]
    base_m = ms["tm_argmax_macro_f1_mean"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for thr in [0.6, 0.7, 0.8]:
        m = [kg["cells"][f"{k:.5g}"][str(thr)]["mean"] for k in kappas]
        s = [kg["cells"][f"{k:.5g}"][str(thr)]["std"] for k in kappas]
        ax.errorbar(kappas, m, yerr=s, fmt="o-", capsize=3,
                    label=fr"$\pi_{{\mathrm{{thr}}}}={thr}$")
    ax.axhline(base_m, color="0.4", ls="--", lw=1, label="base TM")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ (CPSS inverse-regularization strength)")
    ax.set_ylabel("CAS macro-F1 (3 seeds)")
    ax.grid(ls="--", alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/kappa_vs_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_kappa_pi_grid(kg):
    kappas = kg["kappa_grid"]
    M = np.array([[kg["cells"][f"{k:.5g}"][str(t)]["mean"] for t in PI_THRS]
                  for k in kappas])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(PI_THRS)))
    ax.set_xticklabels(PI_THRS)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.3g}" for k in kappas])
    ax.set_xlabel(r"$\pi_{\mathrm{thr}}$")
    ax.set_ylabel(r"$\kappa$")
    for i in range(len(kappas)):
        for j in range(len(PI_THRS)):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() - 0.06 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="CAS macro-F1 (3-seed mean)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/kappa_pi_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def latex_grid_table(kg):
    kappas = kg["kappa_grid"]
    lines = []
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$\kappa \backslash\ \pi_{\mathrm{thr}}$ & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 \\")
    lines.append(r"\midrule")
    for k in kappas:
        row = kg["cells"][f"{k:.5g}"]
        cells = " & ".join(f"{row[str(t)]['mean']:.3f}" for t in PI_THRS)
        lines.append(f"{k:.3g} & {cells} \\\\")
    lines.append(r"\midrule")
    urow = " & ".join(f"{kg['union'][str(t)]['mean']:.3f}" for t in PI_THRS)
    lines.append(r"union grid (CAS) & " + urow + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    print("\n".join(lines))


def main():
    ms, kg = load()
    fig_pi_thr_vs_accuracy(ms)
    fig_kappa_vs_accuracy(kg, ms)
    fig_kappa_pi_grid(kg)
    print("saved pi_thr_vs_accuracy.png, kappa_vs_accuracy.png, kappa_pi_grid.png\n")
    latex_grid_table(kg)


if __name__ == "__main__":
    main()
