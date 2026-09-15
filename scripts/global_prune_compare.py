"""
Uniform per-class budget vs. globally-optimal (knapsack) budget allocation.

Per-class rule : delete k weakest |w| clauses in EVERY class (k identical).
Global rule    : delete the k*C weakest |w| clauses across the whole pool,
                 i.e. the exact greedy solution to
                     max sum_c |D_c|  s.t.  sum_c sum_{j in D_c} |w_cj| <= rho
                 which lets classes with weak clauses give up more.

Both are evaluated on the full 80,868-row test set, 3 seeds.
Output: results/global_prune_compare.json
"""
import json
import pickle

import numpy as np
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NC, NCLS = 400, 6
STEP = 5                       # per-class step -> 30 clauses globally


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    ks = list(range(0, NC, STEP))
    runs = []

    for seed in SEEDS:
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Z = tm.transform(Xte).astype(np.float32)
        W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                            for c in range(NCLS)])
        aw = np.abs(W)

        # per-class ordering (weakest first, within class)
        per_order = {c: c * NC + np.argsort(aw[c * NC:(c + 1) * NC], kind="stable")
                     for c in range(NCLS)}
        # global ordering (weakest first, across whole pool)
        glob_order = np.argsort(aw, kind="stable")

        def evaluate(drop_idx):
            Wm = W.copy()
            Wm[drop_idx] = 0.0
            cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ Wm[c * NC:(c + 1) * NC]
                                  for c in range(NCLS)])
            pred = np.argmax(cs, 1)
            return float(f1_score(yte, pred, average="macro"))

        f_per, f_glob, alloc = [], [], []
        for k in ks:
            drop_p = np.concatenate([per_order[c][:k] for c in range(NCLS)]) \
                if k else np.array([], int)
            f_per.append(evaluate(drop_p))
            ng = k * NCLS
            drop_g = glob_order[:ng] if ng else np.array([], int)
            f_glob.append(evaluate(drop_g))
            kept = [int(NC - np.sum((drop_g >= c * NC) & (drop_g < (c + 1) * NC)))
                    for c in range(NCLS)]
            alloc.append(kept)
        runs.append({"seed": seed, "removed_per_class": ks,
                     "f1_per_class_budget": f_per, "f1_global_budget": f_glob,
                     "global_kept_per_class": alloc})
        print(f"[seed {seed}] done", flush=True)

    out = {"seeds": SEEDS, "removed_per_class": ks, "runs": runs}
    with open(f"{BASE}/global_prune_compare.json", "w") as f:
        json.dump(out, f, indent=2)

    P = np.array([r["f1_per_class_budget"] for r in runs]).mean(0)
    G = np.array([r["f1_global_budget"] for r in runs]).mean(0)
    full = P[0]
    print(f"\nfull-pool macro-F1 {full:.4f}   (epsilon = 0.01)")
    for name, arr in (("per-class budget", P), ("global  budget", G)):
        ok = np.where(arr >= full - 0.01)[0]
        k = ks[ok.max()]
        print(f"  {name}: max removed/class {k}  -> pool {(NC-k)*NCLS:4d} clauses, "
              f"macro-F1 {arr[ok.max()]:.4f}")
    i65 = ks.index(335)
    print(f"\nat the 390-clause pool: per-class {P[i65]:.4f}  global {G[i65]:.4f}")
    print("global allocation at that budget (clauses kept per class):")
    for r in runs:
        print(f"  seed {r['seed']}: {r['global_kept_per_class'][i65]}")


if __name__ == "__main__":
    main()
