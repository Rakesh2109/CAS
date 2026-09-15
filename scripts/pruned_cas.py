"""
CAS on a weight-pruned clause pool.

Step 1 (pruning): within each class, sort the 400 clauses by |learned
weight| and delete the 335 weakest, keeping the top 65 -- the knee of
Fig. clause_pruning (84% of the pool removed for <1 pp macro-F1 loss).
Pool: 6 x 65 = 390 clauses instead of 2,400.

Step 2 (CAS): the original protocol, unchanged -- full 80,868-row test
set, RandomState(0) permutation, first half for CPSS, second half for
evaluation; union over the 6-point kappa grid; sign-split scoring.

Output: results/pruned_cas.json
"""
import json
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
NC, NCLS = 400, 6
KEEP = 65                     # clauses kept per class (335 removed)
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B = 15


def cpss(Z, ybin, seed):
    n, m = Z.shape
    rng = np.random.RandomState(seed)
    half = n // 2
    sel = np.zeros(m, np.int64)
    csum = np.zeros(m, np.float64)
    ng = len(KAPPA_GRID)
    for b in range(B):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            u = np.zeros(m, bool)
            for C in KAPPA_GRID:
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                         max_iter=200, class_weight="balanced",
                                         random_state=0)
                clf.fit(Zh, yh)
                cf = clf.coef_.ravel()
                u |= np.abs(cf) > 1e-8
                csum += cf
            sel += u
    return sel / (2 * B), csum / (2 * B * ng)


def score(Zev, yev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], NCLS))
    sp, sa = [], []
    for i, fam in enumerate(FAMILIES):
        pi, cf = pis[fam], cfs[fam]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        sp.append(len(Sp)); sa.append(int((pi >= thr).sum()))
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    pred = np.argmax(S, 1)
    return (float(f1_score(yev, pred, average="macro")),
            float((pred == yev).mean()), sp, sa)


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Zfull = tm.transform(Xte).astype(np.float32)
        W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                            for c in range(NCLS)])

        # --- step 1: keep top-KEEP clauses per class by |weight| ---
        keep_idx = []
        for c in range(NCLS):
            w_c = W[c * NC:(c + 1) * NC]
            order = np.argsort(np.abs(w_c), kind="stable")   # weakest first
            keep_idx.append(c * NC + np.sort(order[NC - KEEP:]))
        keep_idx = np.concatenate(keep_idx)
        Z = Zfull[:, keep_idx]
        Wk = W[keep_idx]
        m = Z.shape[1]

        # base detector on the pruned pool
        cs = np.column_stack([Z[:, c * KEEP:(c + 1) * KEEP] @ Wk[c * KEEP:(c + 1) * KEEP]
                              for c in range(NCLS)])
        base_pred = np.argmax(cs, 1)

        rng = np.random.RandomState(0)
        perm = rng.permutation(Z.shape[0])
        val, ev = perm[:Z.shape[0] // 2], perm[Z.shape[0] // 2:]
        Zval, Zev, yval, yev = Z[val], Z[ev], yte[val], yte[ev]
        base_f1 = float(f1_score(yev, base_pred[ev], average="macro"))
        base_acc = float((base_pred[ev] == yev).mean())
        print(f"[s{seed}] pool {m} clauses | pruned-base macroF1={base_f1:.4f} "
              f"acc={base_acc:.4f}", flush=True)

        pis, cfs, evb = {}, {}, {}
        for ci, fam in enumerate(FAMILIES):
            tt = time.time()
            pi, cf = cpss(Zval, (yval == ci).astype(int), seed=seed + ci)
            pis[fam], cfs[fam] = pi, cf
            q = float(np.mean(pi) * m)
            evb[fam] = q * q / ((2 * 0.8 - 1) * m)
            print(f"[s{seed}]   CPSS {fam} {time.time()-tt:.0f}s", flush=True)

        out = {"seed": seed, "pool": m, "base_macro_f1": base_f1,
               "base_accuracy": base_acc, "E_V_thr0.8": evb, "pi_thr_sweep": {}}
        for thr in PI_THRS:
            f1, acc, sp, sa = score(Zev, yev, pis, cfs, thr)
            out["pi_thr_sweep"][str(thr)] = {
                "macro_f1": f1, "accuracy": acc,
                "S_pos": sp, "S": sa, "mean_S_pos": float(np.mean(sp))}
            print(f"[s{seed}]   thr={thr}: macroF1={f1:.4f} acc={acc:.4f} "
                  f"mean|S+|={np.mean(sp):.0f}", flush=True)
        runs.append(out)
        print(f"[s{seed}] done {time.time()-t0:.0f}s", flush=True)

    res = {"seeds": SEEDS, "keep_per_class": KEEP, "removed_per_class": NC - KEEP,
           "pool": NCLS * KEEP, "pi_thrs": PI_THRS, "B": B, "runs": runs}
    with open(f"{BASE}/pruned_cas.json", "w") as f:
        json.dump(res, f, indent=2)

    bf = np.array([r["base_macro_f1"] for r in runs])
    print(f"\n=== pruned pool = {NCLS*KEEP} clauses ({KEEP}/class) ===")
    print(f"pruned base TM : macro-F1 {bf.mean():.4f} +/- {bf.std():.4f}")
    for thr in PI_THRS:
        v = np.array([r["pi_thr_sweep"][str(thr)]["macro_f1"] for r in runs])
        a = np.array([r["pi_thr_sweep"][str(thr)]["accuracy"] for r in runs])
        s = np.mean([r["pi_thr_sweep"][str(thr)]["mean_S_pos"] for r in runs])
        print(f"CAS thr={thr}   : macro-F1 {v.mean():.4f} +/- {v.std():.4f}   "
              f"acc {a.mean():.4f}   mean|S+| {s:.0f}")
    print("saved pruned_cas.json")


if __name__ == "__main__":
    main()
