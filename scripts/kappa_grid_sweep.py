"""
kappa (kappa = C = 1/lambda, CPSS inverse-regularization strength) vs accuracy,
and the full kappa x pi_thr 2D grid for CAS attribution on IoTM (CICIoMT2024).

The paper's CPSS uses the UNION over a 6-point geometric kappa grid
(0.001..0.1). This script instruments the identical CPSS fits (same seeds,
same half-fit order, same liblinear random_state) but records the
per-kappa selection counts and mean signed coefficients separately, so we
can report:

  * kappa -> CAS macro-F1 (single-kappa signatures), 3 seeds mean +/- std
  * kappa x pi_thr -> CAS macro-F1 grid, 3 seeds mean +/- std
  * the union-over-grid row (== the paper's operating configuration) for
    reference

No TM retraining: cached tm_model_nbins10_seed{42,7,123}.pkl are reused.
Outputs results/kappa_grid_sweep.json.
"""
import json
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
B = 15
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
MAX_ITER = 200


def cpss_per_kappa(Z, y_bin, seed):
    """CPSS over B complementary pairs, recording per-kappa stats.

    Mirrors stabilis.cpss_proportions exactly (same RNG stream, same fit
    order) but keeps kappa points separate instead of unioning them.
    Returns pi_hat[k] (n_grid, m) and mean_coef[k] (n_grid, m).
    """
    n, m = Z.shape
    n_grid = len(KAPPA_GRID)
    rng = np.random.RandomState(seed)
    sel = np.zeros((n_grid, m), dtype=np.int64)
    coef_sum = np.zeros((n_grid, m), dtype=np.float64)
    union_sel = np.zeros(m, dtype=np.int64)
    union_coef_sum = np.zeros(m, dtype=np.float64)
    half = n // 2

    for b in range(B):
        perm = rng.permutation(n)
        for h_idx in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h_idx], y_bin[h_idx]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            union_this = np.zeros(m, dtype=bool)
            for gi, C in enumerate(KAPPA_GRID):
                clf = LogisticRegression(
                    penalty="l1", solver="liblinear", C=C,
                    max_iter=MAX_ITER, class_weight="balanced",
                    random_state=0,
                )
                clf.fit(Zh, yh)
                coef = clf.coef_.ravel()
                sel[gi] += (np.abs(coef) > 1e-8)
                coef_sum[gi] += coef
                union_this |= (np.abs(coef) > 1e-8)
                union_coef_sum += coef
            union_sel += union_this.astype(np.int64)

    pi_hat = sel / (2 * B)
    mean_coef = coef_sum / (2 * B)
    union_pi = union_sel / (2 * B)
    union_coef = union_coef_sum / (2 * B * n_grid)
    return pi_hat, mean_coef, union_pi, union_coef


def cas_macro_f1(Z_eval, y_eval, pi_hats, mean_coefs, thr):
    """CAS attribution macro-F1 at threshold thr. pi_hats/mean_coefs are
    dicts fam -> (m,) arrays for a single kappa."""
    n = Z_eval.shape[0]
    scores = np.zeros((n, len(FAMILIES)))
    sizes = []
    for i, fam in enumerate(FAMILIES):
        pi_hat, mc = pi_hats[fam], mean_coefs[fam]
        S_pos = np.where((pi_hat >= thr) & (mc > 0))[0]
        S_neg = np.where((pi_hat >= thr) & (mc < 0))[0]
        sizes.append(int(len(S_pos)))
        denom = pi_hat[S_pos].sum() if len(S_pos) else 0.0
        if denom <= 0:
            continue
        pos_term = Z_eval[:, S_pos] @ pi_hat[S_pos] if len(S_pos) else 0.0
        neg_term = Z_eval[:, S_neg] @ pi_hat[S_neg] if len(S_neg) else 0.0
        scores[:, i] = (pos_term - neg_term) / denom
    f1 = f1_score(y_eval, np.argmax(scores, axis=1), average="macro")
    return float(f1), sizes


def run_seed(seed):
    print(f"\n===== seed {seed} =====", flush=True)
    with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    Z = tm.transform(Xte).astype(np.float64)
    rng = np.random.RandomState(0)
    perm = rng.permutation(Z.shape[0])
    val_idx, eval_idx = perm[: Z.shape[0] // 2], perm[Z.shape[0] // 2:]
    Z_val, Z_eval, y_eval = Z[val_idx], Z[eval_idx], yte[eval_idx]

    # per-family CPSS, per-kappa pi_hat / mean_coef
    per_kappa = {gi: {"pi": {}, "coef": {}} for gi in range(len(KAPPA_GRID))}
    union = {"pi": {}, "coef": {}}
    for cls_idx, fam in enumerate(FAMILIES):
        t0 = time.time()
        y_bin = (yte[val_idx] == cls_idx).astype(int)
        pi_hat, mean_coef, union_pi, union_coef = cpss_per_kappa(
            Z_val, y_bin, seed=seed + cls_idx)
        for gi in range(len(KAPPA_GRID)):
            per_kappa[gi]["pi"][fam] = pi_hat[gi]
            per_kappa[gi]["coef"][fam] = mean_coef[gi]
        # true union-over-grid (== the paper's CPSS operating configuration)
        union["pi"][fam] = union_pi
        union["coef"][fam] = union_coef
        print(f"  [{fam}] CPSS {time.time()-t0:.0f}s", flush=True)

    result = {"seed": seed, "kappa_grid": KAPPA_GRID.tolist(),
              "grid": {}, "union": {}}
    for gi, C in enumerate(KAPPA_GRID):
        result["grid"][f"{C:.5g}"] = {}
        for thr in PI_THRS:
            f1, sizes = cas_macro_f1(Z_eval, y_eval, per_kappa[gi]["pi"],
                                     per_kappa[gi]["coef"], thr)
            result["grid"][f"{C:.5g}"][str(thr)] = {
                "cas_macro_f1": f1, "sig_sizes_pos": sizes,
                "mean_sig_size": float(np.mean(sizes))}
            print(f"  kappa={C:.4g} thr={thr}: F1={f1:.4f} "
                  f"mean|S+|={np.mean(sizes):.0f}", flush=True)
    for thr in PI_THRS:
        f1, sizes = cas_macro_f1(Z_eval, y_eval, union["pi"], union["coef"], thr)
        result["union"][str(thr)] = {"cas_macro_f1": f1,
                                     "sig_sizes_pos": sizes,
                                     "mean_sig_size": float(np.mean(sizes))}
    return result


def main():
    runs = [run_seed(s) for s in SEEDS]
    # aggregate kappa x pi_thr
    agg = {"kappa_grid": KAPPA_GRID.tolist(), "pi_thrs": PI_THRS,
           "seeds": SEEDS, "B": B, "runs": runs, "cells": {}}
    for C in KAPPA_GRID:
        key = f"{C:.5g}"
        agg["cells"][key] = {}
        for thr in PI_THRS:
            vals = [r["grid"][key][str(thr)]["cas_macro_f1"] for r in runs]
            szs = [r["grid"][key][str(thr)]["mean_sig_size"] for r in runs]
            agg["cells"][key][str(thr)] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "per_seed": vals, "mean_sig_size": float(np.mean(szs))}
    agg["union"] = {}
    for thr in PI_THRS:
        vals = [r["union"][str(thr)]["cas_macro_f1"] for r in runs]
        agg["union"][str(thr)] = {"mean": float(np.mean(vals)),
                                  "std": float(np.std(vals)), "per_seed": vals}
    with open(f"{BASE}/kappa_grid_sweep.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== kappa x pi_thr (3-seed mean macro-F1) ===")
    hdr = "kappa \\ pi_thr   " + "  ".join(f"{t}" for t in PI_THRS)
    print(hdr)
    for C in KAPPA_GRID:
        row = agg["cells"][f"{C:.5g}"]
        print(f"{C:.5g}".ljust(16) +
              "  ".join(f"{row[str(t)]['mean']:.3f}" for t in PI_THRS))
    print("saved results/kappa_grid_sweep.json")


if __name__ == "__main__":
    main()
