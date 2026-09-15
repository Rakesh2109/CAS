"""
Regenerate the Task A confusion matrices as two separate square PNGs for
single-column stacking in the IEEE paper (Fig. 2). Reads the recorded
primary-run artifact results/task_a_results_seed42_thr0.6.json, so the
numbers are identical to the published two-panel figure.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/FPTM/CAS_IoMT_Empirical"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]

with open(f"{BASE}/results/task_a_results_seed42_thr0.6.json") as f:
    res = json.load(f)

panels = [
    ("tm_argmax", "TM argmax (baseline)", "task_a_cm_baseline.png"),
    ("stabilis", "CAS attribution", "task_a_cm_cas.png"),
]

for key, title, fname in panels:
    cm = np.array(res[key]["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    f1 = res[key]["macro_f1"]

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), FAMILIES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6), FAMILIES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(f"{title}, macro-F1={f1:.3f}", fontsize=10)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalized rate", fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{BASE}/figs/{fname}", dpi=200)
    plt.close(fig)
    print(f"saved figs/{fname}")
