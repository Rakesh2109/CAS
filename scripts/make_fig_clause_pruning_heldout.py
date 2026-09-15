"""Clause-weight pruning curve, partitioned protocol (3-seed mean +/- SD).

Same layout as the earlier clause_pruning figure: report-set macro-F1 and
accuracy against the percentage of clauses removed per class, with the
budget k* chosen on the selection partition marked. Uses the activations
cached by heldout_protocol.py --stage cas.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
OUT = f"{BASE}/heldout"
FIGS = "/FPTM/CAS_IoMT_Empirical/figs"
SEEDS, NC, NCLS, STEP = [42, 7, 123], 400, 6, 5


def main():
    d = json.load(open(f"{OUT}/heldout_cas.json"))
    kstar = d["kstar_removed_per_class"]
    ks = np.array(range(0, NC, STEP))
    yte = np.load(f"{BASE}/booleanized_data.npz")["yte"].astype(int)

    F, A = [], []
    for seed in SEEDS:
        Z = np.asarray(np.load(f"{OUT}/Z_test_seed{seed}.npy"), dtype=np.float32)
        W = np.load(f"{OUT}/W_seed{seed}.npy")
        order = {c: c * NC + np.argsort(np.abs(W[c * NC:(c + 1) * NC]), kind="stable")
                 for c in range(NCLS)}
        f1s, accs = [], []
        for k in ks:
            Wm = W.copy()
            if k:
                Wm[np.concatenate([order[c][:k] for c in range(NCLS)])] = 0.0
            S = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ Wm[c * NC:(c + 1) * NC]
                                 for c in range(NCLS)])
            p = np.argmax(S, 1)
            f1s.append(f1_score(yte, p, average="macro")); accs.append((p == yte).mean())
        F.append(f1s); A.append(accs); del Z
        print(f"[sweep] seed {seed} done", flush=True)

    F, A = np.array(F), np.array(A)
    fm, fs = F.mean(0), F.std(0)
    am, asd = A.mean(0), A.std(0)
    pct = ks / NC * 100

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.fill_between(pct, fm - fs, fm + fs, color="C0", alpha=.18, lw=0)
    ax.plot(pct, fm, "-", color="C0", lw=1.4, marker="o", ms=2.2, label="macro-F1")
    ax.fill_between(pct, am - asd, am + asd, color="C1", alpha=.18, lw=0)
    ax.plot(pct, am, "-", color="C1", lw=1.4, marker="o", ms=2.2, label="accuracy")

    kp = kstar / NC * 100
    ax.axvline(kp, ls="--", lw=1.1, color="0.2")
    ax.annotate(f"$k^*={kstar}$ ({kp:.0f}%)", xy=(kp, ax.get_ylim()[0]),
                xytext=(kp - 1.5, fm.min() + 0.04), rotation=90,
                fontsize=6.5, ha="right", va="bottom", color="0.2")
    ax.set_xlabel("clauses removed per class (%)", fontsize=8)
    ax.set_ylabel("score (report set)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(ls="--", alpha=.3)
    ax.legend(fontsize=7, loc="lower left", framealpha=.9)
    fig.tight_layout(pad=.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/fig_clause_pruning_heldout.{ext}", dpi=300)
    print("saved figs/fig_clause_pruning_heldout.png")
    i = list(ks).index(kstar)
    print(f"  k*={kstar} ({kp:.0f}%): macro-F1 {fm[i]:.4f}+-{fs[i]:.4f}, acc {am[i]:.4f}")
    print(f"  full pool: macro-F1 {fm[0]:.4f}+-{fs[0]:.4f}, acc {am[0]:.4f}")


if __name__ == "__main__":
    main()
