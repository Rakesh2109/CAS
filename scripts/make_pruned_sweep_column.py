"""Compare verified full/pruned sensitivity curves in one paper column."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[1] / "figs"
data = json.loads((FIGS / "fig_sweep_pruned.json").read_text())
experiment = json.loads((FIGS.parent / "results/threeway_cas.json").read_text())
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3,
                     "lines.linewidth": 1.3, "errorbar.capsize": 2,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.80), sharey=True)
for ax, source, xs in ((axes[0], "threshold_sweep", [.5, .6, .7, .8, .9]),
                        (axes[1], "pair_sweep", [5, 10, 15, 20, 25])):
    keys = [f"15|{x}" if source == "threshold_sweep" else f"{x}|0.8" for x in xs]
    for pool, pool_label, linestyle in (("full", "Full", "--"),
                                         ("pruned", "Pruned", "-")):
        rows = [experiment["pools"][pool]["grid"][key] for key in keys]
        for metric, label, color, marker in (("rep_f1", "macro-F1", "C0", "o"),
                                              ("rep_acc", "accuracy", "C1", "s")):
            ax.errorbar(xs, [r[metric] for r in rows],
                        yerr=[r[metric + "_sd"] for r in rows],
                        color=color, marker=marker, ms=3,
                        linestyle=linestyle, markerfacecolor="white" if pool == "full" else color,
                        elinewidth=.7, alpha=.8 if pool == "full" else 1,
                        label=f"{pool_label}: {label}")
    ax.axhline(data["baseline"]["macro_f1"], color="C0", ls=":", lw=1)
    ax.axhline(data["baseline"]["accuracy"], color="C1", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_ylim(.64, .84)
    ax.set_yticks([.65, .70, .75, .80])
    ax.set_ylabel("score (report quarter)", fontsize=7.5)
axes[0].set_title("(a) stability threshold, $B=15$", fontsize=8)
axes[0].set_xlabel("$\\pi_{\\mathrm{thr}}$")
axes[1].set_title("(b) number of pairs, $\\pi_{\\mathrm{thr}}=0.8$", fontsize=8)
axes[1].set_xlabel("complementary pairs $B$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.003),
           ncol=2, frameon=False, fontsize=7, handlelength=2, columnspacing=.9)
fig.tight_layout(pad=.65, h_pad=1, rect=(0,.12,1,1))
for extension in ("pdf", "svg", "png"):
    fig.savefig(FIGS / f"fig_sweep_full_pruned_column.{extension}", dpi=220)
plt.close(fig)
