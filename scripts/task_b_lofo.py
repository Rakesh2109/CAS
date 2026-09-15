"""
Task B -- novel-family flag via Leave-One-Family-Out (LOFO), scoped to 2 of
the 5 possible folds (Spoofing = imbalance stress case per the doc's Ping
Sweep theme, and MQTT = a protocol-specific mid-size family) rather than the
full 5-fold protocol, for tractability in this session.

For each held-out family:
  1. Retrain a restricted multiclass TM on the remaining 5 classes
     (Benign + 4 attack families), the held-out family never seen.
  2. Extract clause activations, build STABILIS signatures (CPSS) on the
     restricted class set.
  3. novelty score nu(x) = 1 - max_l normalized_score_l(x)
  4. Split-conformal (Mondrian, per predicted-family) calibration of nu on
     held-out KNOWN-family calibration flows -> threshold t_l s.t.
     P(false-flag | family=l) <= eps, finite-sample, distribution-free
     (Bates et al. 2023 / Shafer-Vovk exchangeability).
  5. Evaluate: FPR on known test flows (should track eps), TPR on the truly
     unknown held-out family, AUROC/AUPRC (known-vs-unknown) on nu, FPR@95%TPR.
"""
import json
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tmu.models.classification.vanilla_classifier import TMClassifier

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from stabilis import build_signatures, attribution_scores

ALL_FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
LOFO_FOLDS = ["Spoofing", "MQTT"]
EPS = 0.05


def restricted_labels(y, held_out_idx, remaining_families):
    """Remap y (over ALL_FAMILIES indices) to indices over remaining_families
    for rows not belonging to held_out family. Returns mask, y_restricted."""
    mask = y != held_out_idx
    fam_map = {ALL_FAMILIES.index(f): i for i, f in enumerate(remaining_families)}
    y_r = np.array([fam_map[v] for v in y[mask]], dtype=np.uint32)
    return mask, y_r


def run_fold(held_out, Xtr, ytr, Xte, yte, seed=42, clauses=400, T=320, s=8.0, epochs=20, pi_thr=0.6, B=20):
    held_idx = ALL_FAMILIES.index(held_out)
    remaining = [f for f in ALL_FAMILIES if f != held_out]
    print(f"\n=== LOFO fold: held-out = {held_out} | remaining classes = {remaining} ===")

    mask_tr, ytr_r = restricted_labels(ytr, held_idx, remaining)
    Xtr_r = Xtr[mask_tr]

    tm = TMClassifier(number_of_clauses=clauses, T=T, s=s, weighted_clauses=True, platform="CPU", seed=seed)
    t0 = time.time()
    for ep in range(epochs):
        tm.fit(Xtr_r, ytr_r)
    print(f"restricted TM trained in {time.time()-t0:.1f}s on {Xtr_r.shape[0]} rows / {len(remaining)} classes")

    # known-family test rows (remaining families) + unknown-family test rows (held-out)
    mask_known_te = yte != held_idx
    Xte_known, yte_known_all = Xte[mask_known_te], yte[mask_known_te]
    fam_map = {ALL_FAMILIES.index(f): i for i, f in enumerate(remaining)}
    yte_known_r = np.array([fam_map[v] for v in yte_known_all], dtype=np.uint32)
    Xte_unknown = Xte[yte == held_idx]

    Z_known = tm.transform(Xte_known).astype(np.float64)
    Z_unknown = tm.transform(Xte_unknown).astype(np.float64)

    # split known test into: signature-fit / calibration / final-eval (1/3 each)
    rng = np.random.RandomState(1)
    n = Z_known.shape[0]
    perm = rng.permutation(n)
    third = n // 3
    idx_fit, idx_cal, idx_eval = perm[:third], perm[third:2 * third], perm[2 * third:]

    old_families_backup = list(globals()["_orig_families"]) if "_orig_families" in globals() else None
    import stabilis as stab_mod
    stab_mod.FAMILIES = remaining  # restrict signature builder to remaining classes
    signatures, diagnostics = build_signatures(Z_known[idx_fit], yte_known_r[idx_fit], pi_thr=pi_thr, B=B, seed=seed)

    # attribution_scores() already normalizes each family's score to [0,1]
    # (fraction of that family's signature weight present in x)
    scores_cal = attribution_scores(Z_known[idx_cal], signatures, diagnostics)
    scores_eval_known = attribution_scores(Z_known[idx_eval], signatures, diagnostics)
    scores_unknown = attribution_scores(Z_unknown, signatures, diagnostics)

    nu_cal = 1.0 - scores_cal.max(axis=1)
    nu_eval_known = 1.0 - scores_eval_known.max(axis=1)
    nu_unknown = 1.0 - scores_unknown.max(axis=1)

    pred_cal = scores_cal.argmax(axis=1)
    y_cal = yte_known_r[idx_cal]

    # Mondrian (per predicted-family) split-conformal threshold at level EPS
    thresholds = {}
    for i, fam in enumerate(remaining):
        fam_nu = nu_cal[pred_cal == i]
        n_cal = len(fam_nu)
        if n_cal == 0:
            thresholds[fam] = 1.0
            continue
        k = int(np.ceil((n_cal + 1) * (1 - EPS)))
        k = min(k, n_cal)
        thresholds[fam] = float(np.sort(fam_nu)[k - 1])

    pred_eval_known = scores_eval_known.argmax(axis=1)
    thr_vec_known = np.array([thresholds[remaining[p]] for p in pred_eval_known])
    flagged_known = nu_eval_known > thr_vec_known
    fpr_overall = float(flagged_known.mean())

    pred_unknown = scores_unknown.argmax(axis=1)
    thr_vec_unknown = np.array([thresholds[remaining[p]] for p in pred_unknown])
    flagged_unknown = nu_unknown > thr_vec_unknown
    tpr_overall = float(flagged_unknown.mean())

    y_true_ood = np.concatenate([np.zeros(len(nu_eval_known)), np.ones(len(nu_unknown))])
    nu_all = np.concatenate([nu_eval_known, nu_unknown])
    auroc = float(roc_auc_score(y_true_ood, nu_all))
    auprc = float(average_precision_score(y_true_ood, nu_all))
    fpr_curve, tpr_curve, _ = roc_curve(y_true_ood, nu_all)
    idx95 = np.searchsorted(tpr_curve, 0.95)
    fpr_at_95tpr = float(fpr_curve[min(idx95, len(fpr_curve) - 1)])

    per_family_fpr = {}
    for i, fam in enumerate(remaining):
        sel = pred_eval_known == i
        per_family_fpr[fam] = float(flagged_known[sel].mean()) if sel.sum() else None

    print(f"FPR (known, overall) = {fpr_overall:.4f}  (target eps={EPS})")
    print(f"TPR (unknown={held_out}) = {tpr_overall:.4f}")
    print(f"AUROC known-vs-unknown = {auroc:.4f}   AUPRC = {auprc:.4f}   FPR@95%TPR = {fpr_at_95tpr:.4f}")

    result = {
        "held_out_family": held_out,
        "remaining_families": remaining,
        "eps": EPS,
        "thresholds": thresholds,
        "fpr_overall": fpr_overall,
        "tpr_overall": tpr_overall,
        "per_family_fpr": per_family_fpr,
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95tpr": fpr_at_95tpr,
        "n_known_eval": int(len(idx_eval)),
        "n_unknown_eval": int(Xte_unknown.shape[0]),
        "nu_eval_known": nu_eval_known.tolist(),
        "nu_unknown": nu_unknown.tolist(),
    }
    stab_mod.FAMILIES = ALL_FAMILIES  # restore
    return result


def main():
    base = "/FPTM/CAS_IoMT_Empirical/results"
    d = np.load(f"{base}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    all_results = {}
    for held_out in LOFO_FOLDS:
        res = run_fold(held_out, Xtr, ytr, Xte, yte)
        all_results[held_out] = res
        with open(f"{base}/task_b_lofo_{held_out}.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(f"{base}/task_b_lofo_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nsaved all LOFO fold results.")


if __name__ == "__main__":
    main()
