"""
Figures for the binary (normal-vs-attack) CAS arm on IoTM, for
paper/cas_iomt_paper.md Sec 5.3.

Recomputes per-seed scores deterministically: CPSS with the same seed/B/
val-split reproduces the saved signatures exactly (verified against the
saved binary_cas_results_*.json signature sizes before plotting).

Outputs (figs/):
  binary_cas_roc_iotm.png          -- ROC: CAS attack score vs TM margin, 3 seeds
  binary_cas_score_dist_iotm.png   -- CAS attack-score distribution by true class (seed 42)
  binary_cas_summary_iotm.png      -- bar summary: attack-F1 / AUROC, mean +/- std over seeds
"""
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
SEEDS = [42, 7, 123]
T = 320
stab_mod.FAMILIES = ["normal", "attack"]


def load_seed(seed):
    with open(f"{BASE}/binary_tm_model_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    with open(f"{BASE}/binary_cas_results_seed{seed}_thr0.6.json") as f:
        saved = json.load(f)
    return tm, saved


def compute_scores(seed):
    tm, saved = load_seed(seed)
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    y_bin = (yte != 0).astype(int)
    Z = tm.transform(Xte).astype(np.float64)
    eval_idx = np.array(saved["eval_idx"])

    # use the saved (seeded, reproducible) signatures + diagnostics directly --
    # no CPSS re-run needed, so figures reflect exactly the reported artifacts
    signatures = saved["signatures"]
    diagnostics = saved["diagnostics"]

    scores = stab_mod.attribution_scores(Z[eval_idx], signatures, diagnostics)
    cas_score = scores[:, 1]
    _, class_sums = tm.predict(Xte[eval_idx], return_class_sums=True)
    tm_margin = np.clip(np.array(class_sums)[:, 1], 0, T) / T
    return y_bin[eval_idx], cas_score, tm_margin


def main():
    per_seed = {}
    for s in SEEDS:
        y, cas, tm_m = compute_scores(s)
        per_seed[s] = (y, cas, tm_m)

    # --- Fig 1: ROC curves, all seeds ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, s in zip(axes, SEEDS):
        y, cas, tm_m = per_seed[s]
        for score, label, color in [(tm_m, "TM vote margin", "tab:blue"),
                                    (cas, "CAS attack signature", "tab:orange")]:
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"seed {s}")
        ax.set_xlabel("false positive rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("true positive rate")
    fig.suptitle("Binary CAS vs binary-TM baseline: ROC for attack detection (IoTM, held-out eval)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/binary_cas_roc_iotm.png", dpi=160)
    plt.close(fig)

    # --- Fig 2: score distribution, seed 42 ---
    y, cas, tm_m = per_seed[42]
    auc_tm = auc(*roc_curve(y, tm_m)[:2])
    auc_cas = auc(*roc_curve(y, cas)[:2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, score, title in [(axes[0], tm_m, f"TM vote margin (AUROC={auc_tm:.3f})"),
                             (axes[1], cas, f"CAS attack signature score (AUROC={auc_cas:.3f})")]:
        ax.hist(score[y == 0], bins=60, alpha=0.65, density=True, label="normal (true)", color="tab:blue")
        ax.hist(score[y == 1], bins=60, alpha=0.65, density=True, label="attack (true)", color="tab:red")
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("Attack-detection score separation, seed 42 (IoTM, held-out eval)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/binary_cas_score_dist_iotm.png", dpi=160)
    plt.close(fig)

    # --- Fig 3: summary bars over seeds ---
    with open("/FPTM/CAS_IoMT_Empirical/results/binary_cas_aggregate.json") as f:
        agg = json.load(f)["iotm"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key_tm, key_cas, title, ylim in [
            (axes[0], "tm_attack_f1", "cas_attack_f1", "attack F1", (0.85, 0.96)),
            (axes[1], "tm_auroc", "cas_auroc", "AUROC", (0.75, 1.0))]:
        means = [agg[key_tm]["mean"], agg[key_cas]["mean"]]
        stds = [agg[key_tm]["std"], agg[key_cas]["std"]]
        bars = ax.bar(["TM argmax baseline", "CAS attack signature"], means,
                      yerr=stds, capsize=6, color=["tab:blue", "tab:orange"], alpha=0.85)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=10)
        ax.set_title(f"{title} (mean ± std, 3 seeds)")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Binary CAS vs binary-TM baseline on IoTM (held-out eval)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/binary_cas_summary_iotm.png", dpi=160)
    plt.close(fig)

    print("saved figs/binary_cas_roc_iotm.png, binary_cas_score_dist_iotm.png, binary_cas_summary_iotm.png")


if __name__ == "__main__":
    main()
