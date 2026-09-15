"""
New figure: accuracy retained versus signature size.

Plots every CAS operating point (B, pi_thr) as a point in
(signature size, macro-F1) space for both the full and the weight-pruned
clause pool, against the full TM's macro-F1 and its seed band. The message of
the figure is how far the clause pool can be compressed before class-sum
accuracy is measurably affected.

    python3 scripts/make_fig_signature_frontier.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results/heldout"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
ARM = "A_full"
TM_FULL, TM_FULL_SD, TM_FULL_N = 0.6660, 0.0125, 2400
TM_PRUNED, TM_PRUNED_N = 0.6615, 360


def load(fname):
    g = json.load(open(f"{BASE}/{fname}"))[ARM]["grid"]
    size = np.array([v["support"] * 6 for v in g.values()])
    f1 = np.array([v["test_f1"] for v in g.values()])
    o = np.argsort(size)
    return size[o], f1[o], [list(g.keys())[i] for i in o]


def main():
    s_f, y_f, _ = load("split_ablation.json")
    s_p, y_p, k_p = load("split_ablation_pruned.json")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axhspan(TM_FULL - TM_FULL_SD, TM_FULL + TM_FULL_SD,
               color="0.85", zorder=0, label=r"TM $\pm$1 SD")
    ax.axhline(TM_FULL, color="0.35", lw=1.0, ls="--", zorder=1)

    ax.plot(s_f, y_f, "o", ms=3.5, mfc="none", mec="#1f77b4", mew=1.0,
            label="CAS, full pool")
    ax.plot(s_p, y_p, "s", ms=3.5, color="#d62728", label="CAS, pruned pool")
    ax.plot([TM_PRUNED_N], [TM_PRUNED], "^", ms=6, color="#2ca02c",
            label="TM, pruned pool")
    ax.plot([TM_FULL_N], [TM_FULL], "*", ms=10, color="0.2", label="TM, full pool")

    # highlight the pi>=0.8 operating point that matches the full TM exactly
    for key, tag in (("5|0.8", r"$\pi\!\geq\!0.8$"),):
        if key in k_p:
            i = k_p.index(key)
            ax.annotate(tag, (s_p[i], y_p[i]), textcoords="offset points",
                        xytext=(6, -10), fontsize=6.5, color="#d62728")

    ax.set_xscale("log")
    ax.set_xlabel("clause count (pool / positive support)", fontsize=8)
    ax.set_ylabel("macro-F1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(90, 3200)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=5.8, loc="lower right", framealpha=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    out = f"{FIGS}/fig_signature_frontier.png"
    fig.savefig(out, dpi=400)
    print(f"[out] {out}")
    print(f"  full-pool points   : {len(s_f)}  size {s_f.min():.0f}-{s_f.max():.0f}")
    print(f"  pruned-pool points : {len(s_p)}  size {s_p.min():.0f}-{s_p.max():.0f}")


if __name__ == "__main__":
    main()
