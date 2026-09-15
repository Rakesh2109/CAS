"""
Minimax (equal-perturbation) budget allocation -- the exact solution to

    max  sum_c |D_c|   s.t.   max_c  sum_{j in D_c} |w_cj|  <=  rho

Because the argmax decision is protected by 2*max_c R(D_c), the binding
quantity is the WORST class perturbation, not the total. The exact greedy
solution is therefore: in each class independently, delete weakest-first
until that class's cumulative removed |w| would exceed rho. Classes with
many tiny weights give up more clauses; classes with heavy weights give up
fewer. Sweeping rho traces the optimal frontier.

Compared against the uniform per-class budget at matched pool size.
Output: results/minimax_prune.json
"""
import json
import pickle

import numpy as np
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NC, NCLS = 400, 6


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    runs = []

    for seed in SEEDS:
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Z = tm.transform(Xte).astype(np.float32)
        W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                            for c in range(NCLS)])
        aw = np.abs(W)

        order, csum = {}, {}
        for c in range(NCLS):
            o = np.argsort(aw[c * NC:(c + 1) * NC], kind="stable")
            order[c] = o
            csum[c] = np.cumsum(aw[c * NC:(c + 1) * NC][o])   # cumulative removed |w|

        def evaluate(drop_idx):
            Wm = W.copy()
            Wm[drop_idx] = 0.0
            cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ Wm[c * NC:(c + 1) * NC]
                                  for c in range(NCLS)])
            return float(f1_score(yte, np.argmax(cs, 1), average="macro"))

        # sweep rho over a grid spanning the per-class cumulative-weight range
        rho_max = max(csum[c][-1] for c in range(NCLS))
        rhos = np.unique(np.round(np.geomspace(1.0, rho_max, 90)))
        rec = []
        for rho in rhos:
            drop, kept = [], []
            for c in range(NCLS):
                k = int(np.searchsorted(csum[c], rho, side="right"))
                k = min(k, NC - 1)                # never empty a class
                if k:
                    drop.append(c * NC + order[c][:k])
                kept.append(NC - k)
            drop = np.concatenate(drop) if drop else np.array([], int)
            pool = int(sum(kept))
            rec.append({"rho": float(rho), "pool": pool, "kept": kept,
                        "macro_f1": evaluate(drop)})
        runs.append({"seed": seed, "sweep": rec})
        print(f"[seed {seed}] swept {len(rec)} rho values", flush=True)

    with open(f"{BASE}/minimax_prune.json", "w") as f:
        json.dump({"seeds": SEEDS, "runs": runs}, f, indent=2)

    # compare at matched pool sizes against the uniform rule
    U = json.load(open(f"{BASE}/clause_weight_pruning.json"))
    uf = np.array([r["macro_f1"] for r in U["runs"]]).mean(0)
    upool = [(NC - k) * NCLS for k in U["removed_per_class"]]
    full = uf[0]
    print(f"\nfull-pool macro-F1 {full:.4f}, epsilon = 0.01 -> floor {full-0.01:.4f}\n")
    print(f"{'pool':>6} {'uniform':>9} {'minimax':>9}")
    for target in (600, 480, 390, 300, 240, 180, 120):
        ui = int(np.argmin([abs(p - target) for p in upool]))
        mv = []
        for r in runs:
            j = int(np.argmin([abs(s["pool"] - target) for s in r["sweep"]]))
            mv.append(r["sweep"][j]["macro_f1"])
        print(f"{target:>6} {uf[ui]:>9.4f} {np.mean(mv):>9.4f}")

    # smallest pool still within epsilon, per rule
    best = []
    for r in runs:
        ok = [s for s in r["sweep"] if s["macro_f1"] >= full - 0.01]
        best.append(min(s["pool"] for s in ok))
    ok_u = [p for p, v in zip(upool, uf) if v >= full - 0.01]
    print(f"\nsmallest pool within 1 pp -- uniform: {min(ok_u)} clauses; "
          f"minimax: {int(np.mean(best))} clauses (per-seed {best})")


if __name__ == "__main__":
    main()
