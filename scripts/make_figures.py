"""Generate all figures for the CAS/STABILIS empirical run into figs/."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def fig_confusion_matrices(seed=42, pi_thr=0.6):
    with open(f"{BASE}/task_a_results_seed{seed}_thr{pi_thr}.json") as f:
        d = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(axes, ["tm_argmax", "stabilis"], ["TM argmax (baseline)", "STABILIS attribution"]):
        cm = np.array(d[key]["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(FAMILIES))); ax.set_xticklabels(FAMILIES, rotation=45, ha="right")
        ax.set_yticks(range(len(FAMILIES))); ax.set_yticklabels(FAMILIES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title}\nmacro-F1={d[key]['macro_f1']:.3f}")
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.03, label="row-normalized rate")
    fig.suptitle("Task A: closed-set attribution -- confusion matrices (CICIoMT2024 family-level)")
    fig.savefig(f"{FIGS}/task_a_confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_macro_f1_bar(seed=42, pi_thr=0.6):
    with open(f"{BASE}/task_a_results_seed{seed}_thr{pi_thr}.json") as f:
        d = json.load(f)
    fams = FAMILIES
    tm_recall = [d["tm_argmax"]["report"][f]["recall"] for f in fams]
    stab_recall = [d["stabilis"]["report"][f]["recall"] for f in fams]
    x = np.arange(len(fams)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, tm_recall, w, label=f"TM argmax (macro-F1={d['tm_argmax']['macro_f1']:.3f})")
    ax.bar(x + w / 2, stab_recall, w, label=f"STABILIS (macro-F1={d['stabilis']['macro_f1']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("per-family recall"); ax.set_ylim(0, 1)
    ax.set_title("Task A: per-family recall, TM-argmax baseline vs STABILIS attribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(f"{FIGS}/task_a_per_family_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_signature_sizes_and_bound(seed=42, pi_thr=0.6):
    with open(f"{BASE}/stabilis_signatures_seed{seed}_thr{pi_thr}.json") as f:
        d = json.load(f)
    diag = d["diagnostics"]
    fams = FAMILIES
    sizes = [diag[f]["signature_size"] for f in fams]
    bounds = [diag[f]["E_V_bound"] for f in fams]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fams))
    ax1.bar(x, sizes, color="#4C72B0", alpha=0.85, label="|S_l| (signature size)")
    ax1.set_xticks(x); ax1.set_xticklabels(fams, rotation=30, ha="right")
    ax1.set_ylabel("signature size |S_l| (# clauses)")
    ax2 = ax1.twinx()
    ax2.plot(x, bounds, "o-", color="#C44E52", label="E[V] bound (Shah-Samworth)")
    ax2.set_ylabel("E[V] bound (expected # falsely-included clauses)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"STABILIS signature size and error-control certificate (pi_thr={pi_thr})")
    fig.savefig(f"{FIGS}/stabilis_signature_sizes_and_bound.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pi_thr_sweep():
    import pickle, sys
    sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
    import glob
    files = sorted(glob.glob(f"{BASE}/stabilis_signatures_seed42_thr*.json"))
    if len(files) < 2:
        print("skip pi_thr sweep figure (need multiple pi_thr runs)")
        return
    thrs, sizes_by_fam = [], {f: [] for f in FAMILIES}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        thrs.append(d["pi_thr"])
        for fam in FAMILIES:
            sizes_by_fam[fam].append(d["diagnostics"][fam]["signature_size"])
    order = np.argsort(thrs)
    thrs = np.array(thrs)[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in FAMILIES:
        vals = np.array(sizes_by_fam[fam])[order]
        ax.plot(thrs, vals, "o-", label=fam)
    ax.set_xlabel("pi_thr (stability-selection threshold)")
    ax.set_ylabel("signature size |S_l|")
    ax.set_title("Signature size vs stability threshold")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.savefig(f"{FIGS}/pi_thr_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_lofo_novelty():
    import glob
    files = glob.glob(f"{BASE}/task_b_lofo_*.json")
    files = [f for f in files if "all" not in f]
    if not files:
        print("skip LOFO figure (no results yet)")
        return
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, fp in zip(axes, sorted(files)):
        with open(fp) as f:
            d = json.load(f)
        if "nu_eval_known" not in d:
            print("skip novelty-distribution fig (score arrays not in current schema)")
            plt.close(fig)
            return
        nu_known = np.array(d["nu_eval_known"])
        nu_unk = np.array(d["nu_unknown"])
        ax.hist(nu_known, bins=40, alpha=0.6, density=True, label="known families")
        ax.hist(nu_unk, bins=40, alpha=0.6, density=True, label=f"unknown ({d['held_out_family']})")
        ax.set_xlabel("novelty score nu(x)")
        ax.set_ylabel("density")
        ax.set_title(f"LOFO held-out={d['held_out_family']}\nAUROC={d['auroc']:.3f}  "
                      f"FPR={d['fpr_overall']:.3f}  TPR={d['tpr_overall']:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Task B: novelty-score distributions, known vs held-out family (conformal-calibrated)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/task_b_lofo_novelty_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_lofo_metrics_bar():
    import glob
    files = glob.glob(f"{BASE}/task_b_lofo_*.json")
    files = [f for f in files if "all" not in f]
    if not files:
        return
    rows = []
    for fp in sorted(files):
        with open(fp) as f:
            d = json.load(f)
        rows.append(d)
    def _m(r, key):
        # current schema nests metrics under "signature"; older schema was flat
        return r["signature"][key] if "signature" in r else r[key]
    fams = [r["held_out_family"] for r in rows]
    aurocs = [_m(r, "auroc") for r in rows]
    fprs = [_m(r, "fpr_overall") for r in rows]
    tprs = [_m(r, "tpr_overall") for r in rows]
    x = np.arange(len(fams)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w, aurocs, w, label="AUROC (known vs unknown)")
    ax.bar(x, fprs, w, label=f"FPR on known (target eps=0.05)")
    ax.bar(x + w, tprs, w, label="TPR on unknown")
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylim(0, 1)
    ax.set_title("Task B: LOFO novelty-detection summary")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(f"{FIGS}/task_b_lofo_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_stability_kuncheva():
    try:
        with open(f"{BASE}/stability_kuncheva.json") as f:
            d = json.load(f)
    except FileNotFoundError:
        print("skip stability figure (not run yet)")
        return
    fams = FAMILIES
    means = [d["kuncheva"][f]["mean"] if d["kuncheva"][f]["mean"] is not None else 0 for f in fams]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fams, means, color="#55A868")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean pairwise Kuncheva index")
    ax.set_title(f"Signature stability across TM seeds {d['seeds']}")
    ax.set_ylim(-0.2, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(f"{FIGS}/stability_kuncheva.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGS, exist_ok=True)
    fig_confusion_matrices()
    fig_macro_f1_bar()
    fig_signature_sizes_and_bound()
    fig_pi_thr_sweep()
    fig_lofo_novelty()
    fig_lofo_metrics_bar()
    fig_stability_kuncheva()
    print("figures saved to", FIGS)
