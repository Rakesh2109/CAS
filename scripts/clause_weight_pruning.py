"""
Clause-weight pruning ablation on the trained weighted TMs (original
40k/class-capped protocol, seeds 42/7/123), evaluated on the full
80,868-row test set -- i.e. before any CPSS validation/evaluation split.

Within each class the 400 clauses are sorted by |weight| (the weighted TM's
learned integer vote weight). At step k the 5k lowest-|weight| clauses of
every class are removed by zeroing their weight, and the class sums are
recomputed as Z_c @ w_c -- verified to reproduce TMU's own class sums
exactly, so this is an exact ablation, not an approximation.

Output: results/clause_weight_pruning.json
"""
import json
import pickle
import time

import numpy as np
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NC = 400          # clauses per class
NCLS = 6
STEP = 5          # clauses removed per class per step


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    ks = list(range(0, NC, STEP))          # 0,5,...,395 removed per class
    runs = []

    for seed in SEEDS:
        t0 = time.time()
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Z = tm.transform(Xte).astype(np.float32)
        W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                            for c in range(NCLS)])

        # per class: clause indices sorted by |weight| ascending (weakest first)
        order = {}
        for c in range(NCLS):
            w_c = W[c * NC:(c + 1) * NC]
            order[c] = np.argsort(np.abs(w_c), kind="stable")

        f1s, accs, kept = [], [], []
        for k in ks:
            Wm = W.copy()
            for c in range(NCLS):
                drop = order[c][:k]
                Wm[c * NC + drop] = 0.0
            cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ Wm[c * NC:(c + 1) * NC]
                                  for c in range(NCLS)])
            pred = np.argmax(cs, 1)
            f1s.append(float(f1_score(yte, pred, average="macro")))
            accs.append(float((pred == yte).mean()))
            kept.append(NC - k)
        runs.append({"seed": seed, "removed_per_class": ks,
                     "kept_per_class": kept, "macro_f1": f1s, "accuracy": accs})
        print(f"[seed {seed}] full={f1s[0]:.4f}  "
              f"@200 kept={f1s[ks.index(200)]:.4f}  "
              f"@50 kept={f1s[ks.index(350)]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    out = {"seeds": SEEDS, "clauses_per_class": NC, "step": STEP,
           "removed_per_class": ks, "runs": runs}
    with open(f"{BASE}/clause_weight_pruning.json", "w") as f:
        json.dump(out, f, indent=2)

    F = np.array([r["macro_f1"] for r in runs])
    A = np.array([r["accuracy"] for r in runs])
    full = F[:, 0].mean()
    print(f"\nfull-pool macro-F1 {full:.4f}")
    for thresh in (0.01, 0.02, 0.05):
        ok = np.where(F.mean(0) >= full - thresh)[0]
        kmax = ks[ok.max()]
        print(f"  within {thresh*100:.0f} pp of full: up to {kmax} clauses/class "
              f"removed ({kmax/NC*100:.0f}% of pool), "
              f"{NC-kmax} kept, macro-F1 {F.mean(0)[ok.max()]:.4f}")
    print("saved clause_weight_pruning.json")


if __name__ == "__main__":
    main()
