"""Plot pruned-pool sensitivity, matching the full-pool two-panel sweep.

CAS metrics come from threeway_cas.json. The dotted baselines are the
pruned weighted TM on the identical report quarter, recomputed from cached
activations and weights. No training or hyperparameter selection is rerun.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figs"
THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
BS = [5, 10, 15, 20, 25]


def main():
    with (RESULTS / "threeway_cas.json").open() as f:
        data = json.load(f)
    pool = data["pools"]["pruned"]
    grid = pool["grid"]
    with np.load(RESULTS / "booleanized_data.npz") as data_np:
        labels = data_np["yte"]
    n = len(labels)
    report = np.random.RandomState(0).permutation(n)[n // 2 + n // 4:]
    assert len(report) == data["n"]["rep"]
    y = labels[report]
    keep = data["keep_per_class"]
    seeds = [row["seed"] for row in grid["15|0.8"]["per_seed"]]
    baseline_rows = []
    for seed in seeds:
        z = np.load(RESULTS / f"Zte_seed{seed}.npy", mmap_mode="r")
        w = np.load(RESULTS / f"W_seed{seed}.npy")
        classes = len(np.unique(labels))
        nc = len(w) // classes
        sums = []
        for c in range(classes):
            order = np.argsort(np.abs(w[c * nc:(c + 1) * nc]), kind="stable")
            columns = c * nc + np.sort(order[nc - keep:])
            activations = np.asarray(z[np.ix_(report, columns)], dtype=np.float32)
            sums.append(activations @ w[columns])
        pred = np.argmax(np.column_stack(sums), axis=1)
        baseline_rows.append({"seed": seed,
                              "macro_f1": float(f1_score(y, pred, average="macro")),
                              "accuracy": float(np.mean(y == pred))})
    baseline = {key: float(np.mean([r[key] for r in baseline_rows]))
                for key in ("macro_f1", "accuracy")}
    np.testing.assert_allclose(baseline["macro_f1"], pool["base_rep_f1"], atol=1e-12)

    # Verify the saved means and population SDs against the individual seeds.
    for key in {f"15|{t}" for t in THRS} | {f"{b}|0.8" for b in BS}:
        row = grid[key]
        for metric in ("rep_f1", "rep_acc"):
            values = [r[metric] for r in row["per_seed"]]
            np.testing.assert_allclose(row[metric], np.mean(values), atol=1e-12)
            np.testing.assert_allclose(row[metric + "_sd"], np.std(values), atol=1e-12)

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                         "lines.linewidth": 1.4, "errorbar.capsize": 2,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for ax, xs, keys in ((axes[0], THRS, [f"15|{t}" for t in THRS]),
                         (axes[1], BS, [f"{b}|0.8" for b in BS])):
        for metric, label, color, marker in (("rep_f1", "macro-F1", "C0", "o"),
                                              ("rep_acc", "accuracy", "C1", "s")):
            ax.errorbar(xs, [grid[k][metric] for k in keys],
                        yerr=[grid[k][metric + "_sd"] for k in keys],
                        marker=marker, ms=3, label=label, color=color)
        ax.axhline(baseline["macro_f1"], color="C0", ls=":", lw=1)
        ax.axhline(baseline["accuracy"], color="C1", ls=":", lw=1)
        ax.set_xticks(xs)
    axes[0].set_xlabel("$\\pi_{\\mathrm{thr}}$")
    axes[0].set_ylabel("score (report quarter)")
    axes[0].set_title("(a) stability threshold, $B{=}15$", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1].set_xlabel("complementary pairs $B$")
    axes[1].set_title("(b) number of pairs, $\\pi_{\\mathrm{thr}}{=}0.8$", fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"fig_sweep_pruned.{extension}", dpi=200)
    plt.close(fig)
    summary = {"pool": pool["pool"], "keep_per_class": keep,
               "report_rows": len(report), "seeds": seeds,
               "baseline": baseline, "baseline_per_seed": baseline_rows,
               "threshold_sweep": {str(t): grid[f"15|{t}"] for t in THRS},
               "pair_sweep": {str(b): grid[f"{b}|0.8"] for b in BS},
               "caption": "Sensitivity after weight pruning to 390 clauses (65 per class). "
               "Report-quarter mean ± population SD across three seeds. "
               "(a) Threshold sweep at B=15. (b) Pair-count sweep at pi_thr=0.8. "
               "Dotted lines show the pruned weighted-TM macro-F1 and accuracy. "
               "These are sensitivity slices; the selection-quarter operating point "
               "remains B=10, pi_thr=0.9."}
    with (FIGS / "fig_sweep_pruned.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"pool": pool["pool"], "baseline": baseline,
                      "threshold_f1": [grid[f"15|{t}"]["rep_f1"] for t in THRS],
                      "pairs_f1": [grid[f"{b}|0.8"]["rep_f1"] for b in BS]}, indent=2))


if __name__ == "__main__":
    main()
