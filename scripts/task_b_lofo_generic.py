"""
Generic LOFO (leave-one-family-out) novelty-detection runner, for any of the
three datasets. Same protocol as task_b_lofo.py (retrain a restricted TM,
rebuild STABILIS signatures on the restricted class set, Mondrian
split-conformal calibration on known-family flows, evaluate FPR/TPR/AUROC on
held-out known vs the truly-unknown family) -- refactored to take dataset
config instead of hardcoding IoTM's families/paths.
"""
import argparse
import json
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tmu.models.classification.vanilla_classifier import TMClassifier

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod
from stabilis import build_signatures, attribution_scores

EPS = 0.05

DATASET_CONFIG = {
    "iotm": {
        "dir": "results",
        "all_families": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
        "lofo_folds": ["Spoofing", "MQTT"],
        "clauses": 400, "T": 320, "s": 8.0, "epochs": 20,
    },
    "medsec": {
        "dir": "results_medsec",
        "all_families": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
        "lofo_folds": ["Lateral movement", "Exfiltration"],
        "clauses": 400, "T": 320, "s": 8.0, "epochs": 20,
    },
    "wustl": {
        "dir": "results_wustl",
        "all_families": ["normal", "Spoofing", "Data Alteration"],
        "lofo_folds": ["Spoofing", "Data Alteration"],
        "clauses": 200, "T": 160, "s": 8.0, "epochs": 40,
    },
}


def restricted_labels(y, held_out_idx, all_families, remaining_families):
    mask = y != held_out_idx
    fam_map = {all_families.index(f): i for i, f in enumerate(remaining_families)}
    y_r = np.array([fam_map[v] for v in y[mask]], dtype=np.uint32)
    return mask, y_r


def run_fold(held_out, Xtr, ytr, Xte, yte, all_families, clauses, T, s, epochs, seed=42, pi_thr=0.6, B=15):
    held_idx = all_families.index(held_out)
    remaining = [f for f in all_families if f != held_out]
    print(f"\n=== LOFO fold: held-out = {held_out} | remaining classes = {remaining} ===")

    mask_tr, ytr_r = restricted_labels(ytr, held_idx, all_families, remaining)
    Xtr_r = Xtr[mask_tr]

    tm = TMClassifier(number_of_clauses=clauses, T=T, s=s, weighted_clauses=True, platform="CPU", seed=seed)
    t0 = time.time()
    for ep in range(epochs):
        tm.fit(Xtr_r, ytr_r)
    print(f"restricted TM trained in {time.time()-t0:.1f}s on {Xtr_r.shape[0]} rows / {len(remaining)} classes")

    mask_known_te = yte != held_idx
    Xte_known, yte_known_all = Xte[mask_known_te], yte[mask_known_te]
    fam_map = {all_families.index(f): i for i, f in enumerate(remaining)}
    yte_known_r = np.array([fam_map[v] for v in yte_known_all], dtype=np.uint32)
    Xte_unknown = Xte[yte == held_idx]

    Z_known = tm.transform(Xte_known).astype(np.float64)
    Z_unknown = tm.transform(Xte_unknown).astype(np.float64) if Xte_unknown.shape[0] else np.zeros((0, Z_known.shape[1]))

    # TM's own weighted class-vote margin, normalized to [0,1] -- an
    # alternative, information-richer novelty signal to compare against the
    # sparse (~14-60 clause) sign-filtered signature score, since the latter
    # showed uniformly weak power (TPR 3-8%) across every fold tried so far.
    _, cs_known = tm.predict(Xte_known, return_class_sums=True)
    cs_known = np.clip(np.array(cs_known, dtype=np.float64), 0, T) / T
    if Xte_unknown.shape[0]:
        _, cs_unknown = tm.predict(Xte_unknown, return_class_sums=True)
        cs_unknown = np.clip(np.array(cs_unknown, dtype=np.float64), 0, T) / T
    else:
        cs_unknown = np.zeros((0, len(remaining)))

    rng = np.random.RandomState(1)
    n = Z_known.shape[0]
    perm = rng.permutation(n)
    third = n // 3
    idx_fit, idx_cal, idx_eval = perm[:third], perm[third:2 * third], perm[2 * third:]

    stab_mod.FAMILIES = remaining
    signatures, diagnostics = build_signatures(Z_known[idx_fit], yte_known_r[idx_fit], pi_thr=pi_thr, B=B, seed=seed)

    def sig_scores(Z):
        return attribution_scores(Z, signatures, diagnostics)

    score_sources = {
        "signature": (sig_scores(Z_known[idx_cal]), sig_scores(Z_known[idx_eval]), sig_scores(Z_unknown)),
        "tm_margin": (cs_known[idx_cal], cs_known[idx_eval], cs_unknown),
    }

    def conformal_eval(scores_cal, scores_eval_known, scores_unknown):
        nu_cal = 1.0 - scores_cal.max(axis=1)
        nu_eval_known = 1.0 - scores_eval_known.max(axis=1)
        nu_unknown = 1.0 - scores_unknown.max(axis=1) if scores_unknown.shape[0] else np.array([])

        pred_cal = scores_cal.argmax(axis=1)
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

        if len(nu_unknown):
            pred_unknown = scores_unknown.argmax(axis=1)
            thr_vec_unknown = np.array([thresholds[remaining[p]] for p in pred_unknown])
            flagged_unknown = nu_unknown > thr_vec_unknown
            tpr_overall = float(flagged_unknown.mean())
        else:
            tpr_overall = None

        per_family_fpr = {}
        for i, fam in enumerate(remaining):
            sel = pred_eval_known == i
            per_family_fpr[fam] = float(flagged_known[sel].mean()) if sel.sum() else None

        res = {"thresholds": thresholds, "fpr_overall": fpr_overall, "tpr_overall": tpr_overall,
               "per_family_fpr": per_family_fpr,
               "nu_eval_known": nu_eval_known.tolist(), "nu_unknown": nu_unknown.tolist()}

        if len(nu_unknown) and len(set(np.concatenate([np.zeros(len(nu_eval_known)), np.ones(len(nu_unknown))]))) > 1:
            y_true_ood = np.concatenate([np.zeros(len(nu_eval_known)), np.ones(len(nu_unknown))])
            nu_all = np.concatenate([nu_eval_known, nu_unknown])
            res["auroc"] = float(roc_auc_score(y_true_ood, nu_all))
            res["auprc"] = float(average_precision_score(y_true_ood, nu_all))
            fpr_curve, tpr_curve, _ = roc_curve(y_true_ood, nu_all)
            idx95 = np.searchsorted(tpr_curve, 0.95)
            res["fpr_at_95tpr"] = float(fpr_curve[min(idx95, len(fpr_curve) - 1)])
        else:
            res["auroc"] = res["auprc"] = res["fpr_at_95tpr"] = None
        return res

    result = {"held_out_family": held_out, "remaining_families": remaining, "eps": EPS,
              "n_known_eval": int(len(idx_eval)), "n_unknown_eval": int(Xte_unknown.shape[0])}
    for src_name, (sc_cal, sc_eval, sc_unk) in score_sources.items():
        r = conformal_eval(sc_cal, sc_eval, sc_unk)
        result[src_name] = r
        print(f"[{src_name}] FPR={r['fpr_overall']:.4f} (target eps={EPS})  TPR(unknown={held_out})={r['tpr_overall']}"
              + (f"  AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} FPR@95%TPR={r['fpr_at_95tpr']:.4f}" if r["auroc"] is not None else ""))

    stab_mod.FAMILIES = all_families
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASET_CONFIG.keys()))
    args = ap.parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{cfg['dir']}"

    d = np.load(f"{base}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    all_results = {}
    for held_out in cfg["lofo_folds"]:
        res = run_fold(held_out, Xtr, ytr, Xte, yte, cfg["all_families"],
                        cfg["clauses"], cfg["T"], cfg["s"], cfg["epochs"])
        all_results[held_out] = res
        safe_name = held_out.replace(" ", "_")
        with open(f"{base}/task_b_lofo_{safe_name}.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(f"{base}/task_b_lofo_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nsaved all LOFO fold results for", args.dataset)


if __name__ == "__main__":
    main()
