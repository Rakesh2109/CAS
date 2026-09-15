import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

methods = ["TM argmax\n(baseline)", "M1\nSTABILIS", "M2\nSIMPLEXPRO", "Fused\n(M1 + TM)"]
means = [0.9527, 0.8816, 0.7185, 0.9531]
stds = [0.0020, 0.0160, 0.0042, 0.0005]
colors = ["#4C72B0", "#DD8452", "#8172B2", "#55A868"]

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(methods))
ax.bar(x, means, yerr=stds, capsize=6, color=colors)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.015, f"{m:.4f}\n±{s:.4f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("macro-F1 (mean ± std over seeds 42, 7, 123)")
ax.set_ylim(0.6, 1.05)
ax.set_title("MedSec-25: verified across 3 independently-trained TM seeds\n(fusion's seed-42 win over TM-argmax did NOT replicate)")
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig("/FPTM/CAS_IoMT_Empirical/figs/medsec_verified_multiseed.png", dpi=160, bbox_inches="tight")
print("saved medsec_verified_multiseed.png")
