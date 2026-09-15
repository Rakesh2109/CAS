import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical"


def fig_blind_class_repair():
    """WUSTL: Spoofing recall vs lambda, dual-axis with macro-F1 -- names the
    lambda-controlled trade-off between 'best aggregate' and 'best recovery
    of the class the base detector was blind to'."""
    with open(f"{BASE}/results_wustl/fusion_results_seed42_thr0.6.json") as f:
        d = json.load(f)
    lambdas = [row["lambda"] for row in d["lambda_grid"]]
    macro_f1 = [row["val_macro_f1"] for row in d["lambda_grid"]]
    spoof_recall = d["per_class_recall_by_lambda"]["Spoofing"]
    normal_recall = d["per_class_recall_by_lambda"]["normal"]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(lambdas, spoof_recall, "o-", color="#C44E52", label="Spoofing recall (blind class)", linewidth=2)
    ax1.plot(lambdas, normal_recall, "s--", color="#4C72B0", label="normal recall", alpha=0.7)
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel("lambda (fusion weight on STABILIS score)")
    ax1.set_ylabel("recall")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(lambdas, macro_f1, "^:", color="#55A868", label="macro-F1 (aggregate)", linewidth=2)
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0.5, 0.85)

    lam_star = d["selected_lambda"]
    ax1.axvline(lam_star, color="gray", linestyle="--", linewidth=1)
    ax1.text(lam_star, 0.05, f"  macro-F1-optimal\n  lambda={lam_star}", fontsize=8, color="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title("WUSTL-EHMS-2020: 'blind-class repair' -- TM-argmax has\n0.00 Spoofing recall at lambda=0; fusion recovers it as lambda grows")
    fig.tight_layout()
    fig.savefig(f"{BASE}/figs/blind_class_repair_wustl.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved blind_class_repair_wustl.png")


def fig_fusion_summary():
    datasets = [("iotm", "results", "IoTM\n(CICIoMT2024)"), ("medsec", "results_medsec", "MedSec-25\n(kill-chain)"),
                ("wustl", "results_wustl", "WUSTL-EHMS\n(medical IoT)")]
    names, tm_f1, stab_f1, fused_f1, lambdas = [], [], [], [], []
    for key, dirname, label in datasets:
        with open(f"{BASE}/{dirname}/fusion_results_seed42_thr0.6.json") as f:
            d = json.load(f)
        names.append(label)
        tm_f1.append(d["tm_argmax"]["macro_f1"])
        stab_f1.append(d["stabilis_only"]["macro_f1"])
        fused_f1.append(d["fused"]["macro_f1"])
        lambdas.append(d["selected_lambda"])

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w, tm_f1, w, label="TM argmax", color="#4C72B0")
    ax.bar(x, stab_f1, w, label="STABILIS only", color="#DD8452")
    ax.bar(x + w, fused_f1, w, label="Fused (tuned lambda)", color="#55A868")
    for i, lam in enumerate(lambdas):
        ax.text(i + w, fused_f1[i] + 0.015, f"lambda*={lam}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("macro-F1 (eval split)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fusion rule: tm_score + lambda * stabilis_score, lambda tuned per dataset on held-out validation")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{BASE}/figs/fusion_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fusion_summary.png")


if __name__ == "__main__":
    fig_blind_class_repair()
    fig_fusion_summary()
