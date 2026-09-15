import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/FPTM/CAS_IoMT_Empirical"
DATASETS = [("iotm", "results", "IoTM\n(CICIoMT2024)"), ("medsec", "results_medsec", "MedSec-25\n(kill-chain)"),
            ("wustl", "results_wustl", "WUSTL-EHMS\n(medical IoT)")]

names, tm_f1, m1_f1, m2_f1, fused_f1 = [], [], [], [], []
for key, dirname, label in DATASETS:
    with open(f"{BASE}/{dirname}/fusion_results_seed42_thr0.6.json") as f:
        fus = json.load(f)
    with open(f"{BASE}/{dirname}/simplexpro_results_seed42.json") as f:
        sp = json.load(f)
    names.append(label)
    tm_f1.append(fus["tm_argmax"]["macro_f1"])
    m1_f1.append(fus["stabilis_only"]["macro_f1"])
    m2_f1.append(sp["macro_f1"])
    fused_f1.append(fus["fused"]["macro_f1"])

x = np.arange(len(names))
w = 0.2
fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.bar(x - 1.5 * w, tm_f1, w, label="TM argmax (baseline)", color="#4C72B0")
ax.bar(x - 0.5 * w, m1_f1, w, label="M1 STABILIS", color="#DD8452")
ax.bar(x + 0.5 * w, m2_f1, w, label="M2 SIMPLEXPRO", color="#8172B2")
ax.bar(x + 1.5 * w, fused_f1, w, label="M1 fused w/ TM (tuned lambda)", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("macro-F1 (eval split)")
ax.set_ylim(0, 1.05)
ax.set_title("All 3 methods tried x all 3 datasets: TM-argmax vs STABILIS vs SIMPLEXPRO vs fusion")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BASE}/figs/method_comparison_all.png", dpi=160, bbox_inches="tight")
print("saved method_comparison_all.png")

for n, t, s1, s2, fu in zip(names, tm_f1, m1_f1, m2_f1, fused_f1):
    n2 = n.replace(chr(10), " ")
    print(f"{n2:32s} TM={t:.4f}  M1={s1:.4f}  M2={s2:.4f}  fused={fu:.4f}")
