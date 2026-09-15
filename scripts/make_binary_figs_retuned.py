"""
Regenerate the two binary-section figures from the RETUNED binary TM.

Writes the same filenames the manuscript already includes:
  figs/fig_binary_sweep_full_pruned_column.{pdf,png,svg}   (Fig. 4)
  figs/fig_binary_clause_pruning.{pdf,png,svg}             (Fig. 5)
plus figs/fig_binary_clause_pruning.json with the plotted values.

Usage: python3 scripts/make_binary_figs_retuned.py [--tag _c1000t1000]
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/FPTM/CAS_IoMT_Empirical")
BASE = ROOT / "results"
FIGS = ROOT / "figs"
SEEDS = [42, 7, 123]
NCLS = 2

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_c1000t1000")
TAG = ap.parse_args().tag
exp = json.loads((BASE / f"binary_retuned_pruned{TAG}.json").read_text())
NC = exp["config"]["clauses"]

d = np.load(BASE / "booleanized_data.npz")
y = (d["yte"] != 0).astype(int)
perm = np.random.RandomState(0).permutation(len(y))
h, q = len(y) // 2, len(y) // 4
rep = perm[h + q:]


def class_sums(Z, W, cpc):
    return np.column_stack([Z[:, c * cpc:(c + 1) * cpc] @ W[c * cpc:(c + 1) * cpc]
                            for c in range(NCLS)])


# ---------------- Figure: sensitivity sweeps ----------------
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3,
                     "lines.linewidth": 1.3, "errorbar.capsize": 2,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": "black",
                     "axes.labelcolor": "black", "axes.titlecolor": "black",
                     "xtick.color": "black", "ytick.color": "black"})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), sharey=True)
base = exp["pools"]["pruned"]
for ax, source, xs in ((axes[0], "threshold_sweep", [.5, .6, .7, .8, .9]),
                       (axes[1], "pair_sweep", [5, 10, 15, 20, 25])):
    keys = [f"15|{x}" if source == "threshold_sweep" else f"{x}|0.8" for x in xs]
    for pool, plabel, ls in (("full", "Full", "--"), ("pruned", "Pruned", "-")):
        rows = [exp["pools"][pool]["grid"][k] for k in keys]
        for metric, mlabel, color, marker in (("rep_f1", "macro-F1", "C0", "o"),
                                              ("rep_acc", "accuracy", "C1", "s")):
            ax.errorbar(xs, [r[metric] for r in rows], yerr=[r[metric + "_sd"] for r in rows],
                        color=color, marker=marker, ms=3, linestyle=ls,
                        markerfacecolor="white" if pool == "full" else color,
                        elinewidth=.7, alpha=.8 if pool == "full" else 1,
                        label=f"{plabel}: {mlabel}")
    ax.axhline(base["base_rep_f1"], color="C0", ls=":", lw=1)
    ax.axhline(base["base_rep_acc"], color="C1", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_ylabel("score (report quarter)", fontsize=7.5)
lo = min(exp["pools"][p]["grid"][k]["rep_f1"] for p in ("full", "pruned")
         for k in exp["pools"][p]["grid"])
axes[0].set_ylim(min(lo, base["base_rep_f1"]) - .02, .90)
axes[0].set_title("(a) stability threshold, $B=15$", fontsize=8)
axes[0].set_xlabel("$\\pi_{\\mathrm{thr}}$")
axes[1].set_title("(b) number of pairs, $\\pi_{\\mathrm{thr}}=0.8$", fontsize=8)
axes[1].set_xlabel("complementary pairs $B$")
hs, ls_ = axes[0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(.5, .003), ncol=2,
           frameon=False, fontsize=7, handlelength=2, columnspacing=.9)
fig.tight_layout(pad=.65, h_pad=1, rect=(0, .12, 1, 1))
for ext in ("pdf", "svg", "png"):
    fig.savefig(FIGS / f"fig_binary_sweep_full_pruned_column.{ext}", dpi=220)
plt.close(fig)
print("[fig] fig_binary_sweep_full_pruned_column", flush=True)

# ---------------- Figure: clause pruning curve ----------------
ks = np.array(exp["ks"])
f1a, acca, cf1 = [], [], []
for seed in SEEDS:
    Z = np.asarray(np.load(BASE / f"Zte_binary_retuned{TAG}_seed{seed}.npy",
                           mmap_mode="r")[rep], dtype=np.float32)
    W = np.load(BASE / f"W_binary_retuned{TAG}_seed{seed}.npy")
    orders = [c * NC + np.argsort(np.abs(W[c * NC:(c + 1) * NC]), kind="stable")
              for c in range(NCLS)]
    ff, aa, cc = [], [], []
    for k in ks:
        wm = W.copy()
        if k:
            wm[np.concatenate([o[:k] for o in orders])] = 0
        pred = class_sums(Z, wm, NC).argmax(1)
        csc = f1_score(y[rep], pred, labels=[0, 1], average=None, zero_division=0)
        cc.append(csc.tolist()); ff.append(float(csc.mean()))
        aa.append(float((y[rep] == pred).mean()))
    f1a.append(ff); acca.append(aa); cf1.append(cc)
f1a, acca, cf1 = np.array(f1a), np.array(acca), np.array(cf1)
assert np.allclose(f1a, np.array(exp["prune_curve_rep"]), atol=1e-12), \
    "recomputed prune curve disagrees with the saved one"
kstar = exp["kstar_removed_per_class"]
smean = np.array(exp["prune_curve_sel"]).mean(0)
assert kstar == ks[np.where(smean >= smean[0] - .01)[0][-1]], "k* rule mismatch"
ki = list(ks).index(kstar)
(FIGS / "fig_binary_clause_pruning.json").write_text(json.dumps(
    {"config": exp["config"], "seeds": SEEDS, "ks": ks.tolist(), "selected_k": kstar,
     "keep_per_class": exp["keep_per_class"],
     "budget_selection_partition": "selection quarter", "plot_partition": "report quarter",
     "macro_f1_per_seed": f1a.tolist(), "accuracy_per_seed": acca.tolist(),
     "class_order": ["Benign", "Attack"], "class_f1_per_seed": cf1.tolist()}, indent=2) + "\n")

plt.rcParams.update({"font.size": 12, "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(5.0, 3.3))
for vals, color, label, style in [(f1a, "C1", "macro-F1", "-"), (acca, "C0", "accuracy", "-"),
                                  (cf1[:, :, 0], "C2", "Benign F1", "--"),
                                  (cf1[:, :, 1], "C3", "Attack F1", "-.")]:
    m, sd = vals.mean(0), vals.std(0)
    ax.fill_between(ks, m - sd, m + sd, color=color, alpha=.18)
    ax.plot(ks, m, color=color, lw=1.6, ls=style, label=label)
ax.axhline(f1a.mean(0)[0], ls="--", lw=1, color=".45", label="full-pool macro-F1")
ax.axvline(kstar, ls=":", lw=1.2, color=".25")
ax.annotate(f"$k^*={kstar}$ removed/class\n{exp['keep_per_class']} retained/class",
            xy=(kstar, f1a.mean(0)[ki]), xytext=(NC * 0.30, .30), fontsize=10,
            arrowprops={"arrowstyle": "->", "lw": .8, "color": ".25"})
ax.set_xlabel("clauses removed per class")
ax.set_ylabel("report-quarter score")
ax.set_xlim(0, NC); ax.set_ylim(0, 1)
ax.grid(ls="--", alpha=.3)
ax.legend(fontsize=9, loc="lower left", ncol=2, columnspacing=1, handlelength=2)
fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(FIGS / f"fig_binary_clause_pruning.{ext}", dpi=220)
plt.close(fig)
print("[fig] fig_binary_clause_pruning", flush=True)
print(f"k=0   macro-F1 {f1a.mean(0)[0]:.4f}+/-{f1a.std(0)[0]:.4f} acc {acca.mean(0)[0]:.4f}+/-{acca.std(0)[0]:.4f}")
print(f"k={kstar} macro-F1 {f1a.mean(0)[ki]:.4f}+/-{f1a.std(0)[ki]:.4f} acc {acca.mean(0)[ki]:.4f}+/-{acca.std(0)[ki]:.4f}")
print(f"k=0   class F1 {cf1[:,0,:].mean(0)} ; k={kstar} class F1 {cf1[:,ki,:].mean(0)}")

# Keep manuscript-local PDF copies synchronized; TeX resolves them before graphicspath.
import shutil
for filename in ("fig_binary_sweep_full_pruned_column.pdf", "fig_binary_clause_pruning.pdf"):
    shutil.copy2(FIGS / filename, ROOT / "IEEE-conference-template-062824 2" / filename)
