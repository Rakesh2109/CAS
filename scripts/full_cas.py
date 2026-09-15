"""
CAS analysis on the FULL-dataset trained TMs.

Base-detector metrics (Task A macro-F1 multiclass; F1/R/P/AUROC binary) are
computed on the FULL 2.18M-row test set. The CPSS signature analysis
(pi_thr sweep, B sweep, kappa x pi_thr grid, signature compaction,
disjunct certificate) runs on a stratified test subsample -- L1-logistic
CPSS on 2M rows is intractable, and a ~10k/class subsample is ample for
stability selection.

  python3 scripts/full_cas.py --arm multiclass --seeds 42,7,123
  python3 scripts/full_cas.py --arm binary     --seeds 42,7,123
"""
import argparse
import json
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAM_MC = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
FAM_BIN = ["normal", "attack"]
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAP = [5, 10, 15, 20, 25]
B_MAX = 25
CAP_PER_CLASS = 12000
SUBSEED = 0


def stratified_subsample(y, cap, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        take = ci if len(ci) <= cap else rng.choice(ci, cap, replace=False)
        idx.append(take)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def cpss_full(Z, ybin, seed, b_max=B_MAX):
    """One CPSS pass, B=b_max complementary pairs. Records, per kappa point,
    the selection count and signed-coef sum after each B in B_SNAP, plus the
    union-over-grid selection. Returns nested dict keyed by (B, 'union'|kappa_i)."""
    n, m = Z.shape
    ng = len(KAPPA_GRID)
    rng = np.random.RandomState(seed)
    half = n // 2
    # accumulators
    sel = np.zeros((ng, m), np.int64)
    coef = np.zeros((ng, m), np.float64)
    usel = np.zeros(m, np.int64)
    ucoef = np.zeros(m, np.float64)
    snaps = {}
    hf = 0  # completed half-fits
    for b in range(1, b_max + 1):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            hf += 1
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
            d = 2 * b
            snaps[b] = {
                "union": (usel / d, ucoef / (d * ng)),
                "per_kappa": [(sel[gi] / d, coef[gi] / d) for gi in range(ng)],
            }
    return snaps


def cas_scores(Zev, pi, coef, thr):
    Spos = np.where((pi >= thr) & (coef > 0))[0]
    Sneg = np.where((pi >= thr) & (coef < 0))[0]
    den = pi[Spos].sum()
    if den <= 0:
        return None, len(Spos), len(Sneg)
    s = (Zev[:, Spos] @ pi[Spos] - (Zev[:, Sneg] @ pi[Sneg] if len(Sneg) else 0)) / den
    return s, len(Spos), len(Sneg)


def ev_bound(pi, thr):
    p = len(pi)
    q = float(np.mean(pi) * p)
    return q * q / ((2 * thr - 1) * p) if thr > 0.5 else float("inf")


def run_seed(arm, seed, prefix="full"):
    fam = FAM_MC if arm == "multiclass" else FAM_BIN
    K = len(fam)
    nb = 10 if arm == "multiclass" else 5
    d = np.load(f"{BASE}/{prefix}_bool_nbins{nb}.npz")
    Xte, yte = d["Xte"], d["yte"]
    if arm == "binary":
        yte = (yte != 0).astype(np.uint8)
    with open(f"{BASE}/tm_{prefix}_{arm}_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)

    # --- base detector on FULL test ---
    t0 = time.time()
    pred, cs = tm.predict(np.ascontiguousarray(Xte, np.uint32), return_class_sums=True)
    cs = np.asarray(cs, np.float64)
    base = {"macro_f1": float(f1_score(yte, pred, average="macro")),
            "accuracy": float((pred == yte).mean())}
    if arm == "binary":
        T = tm.T
        margin = np.clip(cs[:, 1] - cs[:, 0], -T, T) / (2 * T) + 0.5
        base.update(attack_f1=float(f1_score(yte, pred, pos_label=1)),
                    attack_recall=float(recall_score(yte, pred, pos_label=1)),
                    attack_precision=float(precision_score(yte, pred, pos_label=1)),
                    auroc=float(roc_auc_score(yte, margin)),
                    auprc=float(average_precision_score(yte, margin)))
    print(f"[{arm} s{seed}] base {base}  ({time.time()-t0:.0f}s)", flush=True)

    # --- CPSS subsample ---
    sub = stratified_subsample(yte, CAP_PER_CLASS, SUBSEED)
    Zsub = tm.transform(np.ascontiguousarray(Xte[sub], np.uint32)).astype(np.float32)
    ysub = yte[sub]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(sub))
    val, ev = perm[:len(sub) // 2], perm[len(sub) // 2:]
    Zval, Zev, yval, yev = Zsub[val], Zsub[ev], ysub[val], ysub[ev]
    print(f"[{arm} s{seed}] CPSS subsample {Zsub.shape} val {len(val)} eval {len(ev)}", flush=True)

    # CPSS per family
    fam_snaps = {}
    for ci, fname in enumerate(fam):
        tt = time.time()
        fam_snaps[fname] = cpss_full(Zval, (yval == ci).astype(int), seed=seed + ci)
        print(f"[{arm} s{seed}] CPSS {fname} {time.time()-tt:.0f}s", flush=True)

    # --- B sweep (union grid) x pi_thr ---
    B_sweep = {}
    for B in B_SNAP:
        row = {}
        for thr in PI_THRS:
            scores = np.full((len(ev), K), -1e9)
            szpos = []
            for ci, fname in enumerate(fam):
                pi, cf = fam_snaps[fname][B]["union"]
                s, npos, nneg = cas_scores(Zev, pi, cf, thr)
                szpos.append(npos)
                if s is not None:
                    scores[:, ci] = s
            f1 = float(f1_score(yev, np.argmax(scores, 1), average="macro"))
            acc = float((np.argmax(scores, 1) == yev).mean())
            row[str(thr)] = {"macro_f1": f1, "accuracy": acc,
                             "mean_sig_pos": float(np.mean(szpos))}
        B_sweep[B] = row

    # --- kappa x pi_thr grid (B=15) ---
    kap = {}
    for gi, C in enumerate(KAPPA_GRID):
        kap[f"{C:.5g}"] = {}
        for thr in PI_THRS:
            scores = np.full((len(ev), K), -1e9)
            for ci, fname in enumerate(fam):
                pi, cf = fam_snaps[fname][15]["per_kappa"][gi]
                s, *_ = cas_scores(Zev, pi, cf, thr)
                if s is not None:
                    scores[:, ci] = s
            kap[f"{C:.5g}"][str(thr)] = float(
                f1_score(yev, np.argmax(scores, 1), average="macro"))

    # --- signature compaction table (B=15, pi_thr=0.8) ---
    compaction = {}
    for ci, fname in enumerate(fam):
        pi, cf = fam_snaps[fname][15]["union"]
        S = int((pi >= 0.8).sum())
        Spos = int(((pi >= 0.8) & (cf > 0)).sum())
        compaction[fname] = {"orig": int(len(pi)), "S": S, "S_pos": Spos,
                             "E_V": round(ev_bound(pi, 0.8), 1)}

    # --- CAS attribution headline (B=15, pi_thr 0.8) with per-class + binary metrics ---
    thr = 0.8
    scores = np.full((len(ev), K), -1e9)
    for ci, fname in enumerate(fam):
        pi, cf = fam_snaps[fname][15]["union"]
        s, *_ = cas_scores(Zev, pi, cf, thr)
        if s is not None:
            scores[:, ci] = s
    cas_pred = np.argmax(scores, 1)
    cas = {"macro_f1": float(f1_score(yev, cas_pred, average="macro")),
           "accuracy": float((cas_pred == yev).mean())}
    if arm == "binary":
        cas_score = scores[:, 1] - scores[:, 0]
        cas.update(attack_f1=float(f1_score(yev, cas_pred, pos_label=1)),
                   attack_recall=float(recall_score(yev, cas_pred, pos_label=1)),
                   attack_precision=float(precision_score(yev, cas_pred, pos_label=1)),
                   auroc=float(roc_auc_score(yev, cas_score)),
                   auprc=float(average_precision_score(yev, cas_score)))

    # --- disjunct certificate on the inculpatory signatures (B=15, pi_thr 0.6 & 0.8) ---
    disj = {}
    if arm == "multiclass":
        import sys
        sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
        import disjunct as DJ
        for pt in (0.6, 0.8):
            sig = {}
            for ci, fname in enumerate(fam):
                pi, cf = fam_snaps[fname][15]["union"]
                sig[fname] = {"pos": np.where((pi >= pt) & (cf > 0))[0].tolist()}
            m = len(fam_snaps[fam[0]][15]["union"][0])
            Bmat = DJ.build_incidence_matrix(sig, fam, m)
            that, covers = DJ.t_disjunct_certificate(Bmat, fam)
            Bred, nrem = DJ.greedy_overlap_reduction(Bmat, fam)
            thatr, _ = DJ.t_disjunct_certificate(Bred, fam)
            row = {"t_hat": int(that), "t_hat_reduced": int(thatr),
                   "n_removed": int(nrem),
                   "sizes": {f: int(Bmat[:, i].sum()) for i, f in enumerate(fam)}}
            for npz, ct in [(0.0, 1.0), (0.05, 1.0), (0.05, 0.9), (0.05, 0.8), (0.15, 0.8)]:
                acc, _ = DJ.simulate_mixture_decoding(Bmat, fam, t=2, n_trials=500,
                                                      seed=seed, noise_p=npz,
                                                      comp_threshold=ct)
                row[f"mix_noise{npz}_thr{ct}"] = acc
            disj[str(pt)] = row

    return {"seed": seed, "arm": arm, "base": base, "cas_thr0.8_B15": cas,
            "B_sweep": B_sweep, "kappa_grid": kap, "compaction": compaction,
            "disjunct": disj,
            "n_test_full": int(len(yte)), "n_eval_sub": int(len(ev))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["multiclass", "binary"], required=True)
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--prefix", default="full")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    runs = []
    for sd in seeds:
        try:
            runs.append(run_seed(args.arm, sd, args.prefix))
        except FileNotFoundError as e:
            print(f"skip seed {sd}: {e}", flush=True)
    out = {"arm": args.arm, "seeds": seeds, "kappa_grid": KAPPA_GRID.tolist(),
           "pi_thrs": PI_THRS, "B_snap": B_SNAP, "runs": runs}
    with open(f"{BASE}/{args.prefix}_cas_{args.arm}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {BASE}/{args.prefix}_cas_{args.arm}.json", flush=True)


if __name__ == "__main__":
    main()
