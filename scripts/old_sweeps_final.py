"""
Sensitivity sweeps for the ORIGINAL paper protocol (40k/class-capped
booleanized_data.npz, tm_model_nbins10_seed{42,7,123}.pkl).

Mirrors iotm_multiseed_perclass.py exactly -- full 80,868-row test set,
RandomState(0) permutation, first half = CPSS validation, second half =
evaluation -- so every number here is directly comparable with Table I.

Records, per seed:
  * union-over-kappa-grid pi_hat snapshots after B in {5,10,15,20,25}
  * per-kappa pi_hat at B=15 (for the kappa x pi_thr grid)
  * CAS macro-F1 AND accuracy at pi_thr in {0.5..0.9} for every snapshot

Output: results/old_sweeps_final.json
"""
import json
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAP = [5, 10, 15, 20, 25]
B_MAX = 25


def cpss(Z, ybin, seed):
    """CPSS with per-kappa and union bookkeeping, snapshotted at each B."""
    n, m = Z.shape
    ng = len(KAPPA_GRID)
    rng = np.random.RandomState(seed)
    half = n // 2
    sel = np.zeros((ng, m), np.int64)
    coef = np.zeros((ng, m), np.float64)
    usel = np.zeros(m, np.int64)
    ucoef = np.zeros(m, np.float64)
    snaps = {}
    for b in range(1, B_MAX + 1):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            uthis = np.zeros(m, bool)
            for gi, C in enumerate(KAPPA_GRID):
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                         max_iter=200, class_weight="balanced",
                                         random_state=0)
                clf.fit(Zh, yh)
                cf = clf.coef_.ravel()
                nz = np.abs(cf) > 1e-8
                sel[gi] += nz
                coef[gi] += cf
                uthis |= nz
                ucoef += cf
            usel += uthis
        if b in B_SNAP:
            dd = 2 * b
            snaps[b] = {"union": (usel / dd, ucoef / (dd * ng)),
                        "per_kappa": [(sel[gi] / dd, coef[gi] / dd) for gi in range(ng)]}
    return snaps


def score(Zev, yev, pis, coefs, thr):
    n = Zev.shape[0]
    S = np.zeros((n, len(FAMILIES)))
    sizes = []
    for i, fam in enumerate(FAMILIES):
        pi, cf = pis[fam], coefs[fam]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        sizes.append(len(Sp))
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    pred = np.argmax(S, 1)
    return (float(f1_score(yev, pred, average="macro")),
            float((pred == yev).mean()), float(np.mean(sizes)))


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Z = tm.transform(Xte).astype(np.float32)
        rng = np.random.RandomState(0)
        perm = rng.permutation(Z.shape[0])
        val, ev = perm[:Z.shape[0] // 2], perm[Z.shape[0] // 2:]
        Zval, Zev, yval, yev = Z[val], Z[ev], yte[val], yte[ev]
        tm_pred = tm.predict(Xte)
        base_f1 = float(f1_score(yte[ev], tm_pred[ev], average="macro"))
        base_acc = float((tm_pred[ev] == yte[ev]).mean())
        print(f"[s{seed}] base macroF1={base_f1:.4f} acc={base_acc:.4f} "
              f"| val {len(val)} eval {len(ev)}", flush=True)

        snaps = {}
        for ci, fam in enumerate(FAMILIES):
            tt = time.time()
            snaps[fam] = cpss(Zval, (yval == ci).astype(int), seed=seed + ci)
            print(f"[s{seed}] CPSS {fam} {time.time()-tt:.0f}s", flush=True)

        out = {"seed": seed, "base_macro_f1": base_f1, "base_accuracy": base_acc,
               "B_sweep": {}, "kappa_grid": {}}
        for B in B_SNAP:
            out["B_sweep"][str(B)] = {}
            pis = {f: snaps[f][B]["union"][0] for f in FAMILIES}
            cfs = {f: snaps[f][B]["union"][1] for f in FAMILIES}
            for thr in PI_THRS:
                f1, acc, sz = score(Zev, yev, pis, cfs, thr)
                out["B_sweep"][str(B)][str(thr)] = {
                    "macro_f1": f1, "accuracy": acc, "mean_sig_pos": sz}
        for gi, C in enumerate(KAPPA_GRID):
            key = f"{C:.5g}"
            out["kappa_grid"][key] = {}
            pis = {f: snaps[f][15]["per_kappa"][gi][0] for f in FAMILIES}
            cfs = {f: snaps[f][15]["per_kappa"][gi][1] for f in FAMILIES}
            for thr in PI_THRS:
                f1, acc, sz = score(Zev, yev, pis, cfs, thr)
                out["kappa_grid"][key][str(thr)] = {"macro_f1": f1, "accuracy": acc}
        runs.append(out)
        print(f"[s{seed}] done in {time.time()-t0:.0f}s", flush=True)

    res = {"seeds": SEEDS, "pi_thrs": PI_THRS, "B_snap": B_SNAP,
           "kappa_grid": KAPPA_GRID.tolist(), "runs": runs}
    with open(f"{BASE}/old_sweeps_final.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved old_sweeps_final.json", flush=True)


if __name__ == "__main__":
    main()
