import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical"
DATASETS = [
    ("IoTM (CICIoMT2024, family-level)", "results/task_a_results_seed42_thr0.6.json", "results/tm_argmax_baseline_seed42.json"),
    ("MedSec-25 (kill-chain stages)", "results_medsec/task_a_results_seed42_thr0.6.json", "results_medsec/tm_argmax_baseline_seed42.json"),
    ("WUSTL-EHMS-2020 (medical IoT)", "results_wustl/task_a_results_seed42_thr0.6.json", "results_wustl/tm_argmax_baseline_seed42.json"),
]

names, tm_f1, stab_f1, base_f1 = [], [], [], []
for name, task_a_path, base_path in DATASETS:
    with open(f"{BASE}/{task_a_path}") as f:
        ta = json.load(f)
    with open(f"{BASE}/{base_path}") as f:
        base = json.load(f)
    names.append(name)
    tm_f1.append(ta["tm_argmax"]["macro_f1"])
    stab_f1.append(ta["stabilis"]["macro_f1"])
    base_f1.append(base["macro_f1"])

x = np.arange(len(names))
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - w / 2, tm_f1, w, label="TM argmax (baseline)", color="#4C72B0")
ax.bar(x + w / 2, stab_f1, w, label="STABILIS attribution", color="#DD8452")
ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, label="doc's closed-set sanity gate (0.85)")
for i, (a, b) in enumerate(zip(tm_f1, stab_f1)):
    ax.text(i - w / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10, ha="center", fontsize=9)
ax.set_ylabel("macro-F1 (closed-set attribution)")
ax.set_ylim(0, 1.08)
ax.set_title("Cross-dataset comparison: which IoT/IoMT dataset is 'best' for CAS/STABILIS?")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BASE}/figs/cross_dataset_comparison.png", dpi=160, bbox_inches="tight")
print("saved cross_dataset_comparison.png")

for n, t, s in zip(names, tm_f1, stab_f1):
    print(f"{n:45s} TM-argmax={t:.4f}  STABILIS={s:.4f}  delta={s-t:+.4f}")
