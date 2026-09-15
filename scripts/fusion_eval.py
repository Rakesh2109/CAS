"""
Fusion rule: combine the TM's own weighted class-vote (class_sum, normalized
by T) with the STABILIS signature match score, instead of treating STABILIS
as a standalone competing classifier.

combined_l(x) = tm_score_l(x) + lambda * stabilis_score_l(x)
  tm_score_l(x)      = clip(class_sum_l(x), 0, T) / T           in [0,1]
  stabilis_score_l(x)= attribution_scores(...)                   in [0,1]
pred(x) = argmax_l combined_l(x)

lambda is chosen per dataset by a small grid search on the CPSS validation
split (val_idx, disjoint from eval_idx -- no leakage), then evaluated once on
eval_idx. Hypothesis under test: on a strong base detector (MedSec, WUSTL)
lambda should shrink toward 0 (fusion ~= argmax), while on a weak/noisy base
detector (IoTM) a larger lambda should still help, recovering close to the
standalone-STABILIS gain measured earlier.
"""
import argparse
import json
import pickle
import sys
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod

DATASET_FAMILIES = {
    "iotm": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
    "medsec": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
    "wustl": ["normal", "Spoofing", "Data Alteration"],
}
DATASET_DIRS = {"iotm": "results", "medsec": "results_medsec", "wustl": "results_wustl"}
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0]


def tm_scores_from_class_sums(class_sums, T):
    cs = np.clip(class_sums, 0, T).astype(np.float64) / T
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["iotm", "medsec", "wustl"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi_thr", type=float, default=0.6)
    args = ap.parse_args()

    families = DATASET_FAMILIES[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{DATASET_DIRS[args.dataset]}"
    stab_mod.FAMILIES = families

    with open(f"{base}/tm_model_seed{args.seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{base}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    with open(f"{base}/stabilis_signatures_seed{args.seed}_thr{args.pi_thr}.json") as f:
        sig_data = json.load(f)
    signatures, diagnostics = sig_data["signatures"], sig_data["diagnostics"]
    val_idx = np.array(sig_data["val_idx"])
    eval_idx = np.array(sig_data["eval_idx"])

    print("computing Z, class_sums...")
    t0 = time.time()
    Z = tm.transform(Xte).astype(np.float64)
    tm_argmax_preds_true, class_sums = tm.predict(Xte, return_class_sums=True)
    class_sums = np.array(class_sums, dtype=np.float64)
    # tm_argmax_preds_true is the REAL prediction (tm.predict's own internal
    # argmax on raw, unclipped class_sums). Clipping class_sums to [0,T] for
    # the fusion score below is legitimate (bounds each class's contribution
    # to [0,1] for the additive formula) but must NOT be used to derive the
    # "TM-argmax baseline" comparison number: clip-then-argmax silently
    # differs from true argmax whenever multiple classes have negative raw
    # sums (clipping creates ties argmax breaks arbitrarily) -- verified this
    # flips 18/13076 predictions on MedSec seed 42, a real discrepancy against
    # task_a_eval.py's/train_tm.py's baseline number, not noise.
    print(f"done in {time.time()-t0:.1f}s")

    T = tm.T
    tm_score_all = tm_scores_from_class_sums(class_sums, T)
    stab_score_all = stab_mod.attribution_scores(Z, signatures, diagnostics)

    y_val, y_eval = yte[val_idx], yte[eval_idx]
    tm_val, stab_val = tm_score_all[val_idx], stab_score_all[val_idx]
    tm_eval, stab_eval = tm_score_all[eval_idx], stab_score_all[eval_idx]

    # pick lambda on validation split -- report the full macro-F1 AND
    # per-class-recall trade-off curve, not just the macro-F1 argmax, since
    # macro-F1-optimal lambda can sacrifice recall on the class that matters
    # most forensically (the one the base detector was blind to).
    best_lambda, best_f1 = 0.0, -1
    lambda_curve = []
    per_class_recall_by_lambda = {fam: [] for fam in families}
    for lam in LAMBDA_GRID:
        combined = tm_val + lam * stab_val
        preds = combined.argmax(axis=1)
        f1 = f1_score(y_val, preds, average="macro")
        rep = classification_report(y_val, preds, target_names=families, output_dict=True, zero_division=0)
        for fam in families:
            per_class_recall_by_lambda[fam].append(rep[fam]["recall"])
        lambda_curve.append({"lambda": lam, "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_lambda = f1, lam
    print("lambda sweep (validation):", lambda_curve)
    print(f"selected lambda (macro-F1-optimal) = {best_lambda} (val macro-F1={best_f1:.4f})")
    print("per-class recall vs lambda:")
    for fam in families:
        print(f"  {fam:20s}", [round(v, 3) for v in per_class_recall_by_lambda[fam]])

    # baselines on eval split for reference. tm_only_preds uses the TRUE
    # tm.predict() argmax (not clip-then-argmax on tm_score, which can flip
    # ties among negative-raw-sum classes -- see note above).
    tm_only_preds = tm_argmax_preds_true[eval_idx]
    stab_only_preds = stab_eval.argmax(axis=1)
    fused_preds = (tm_eval + best_lambda * stab_eval).argmax(axis=1)

    def report(preds, name):
        f1 = f1_score(y_eval, preds, average="macro")
        rep = classification_report(y_eval, preds, target_names=families, output_dict=True)
        cm = confusion_matrix(y_eval, preds, labels=list(range(len(families))))
        print(f"{name:20s} macro-F1={f1:.4f}")
        return {"macro_f1": f1, "report": rep, "confusion_matrix": cm.tolist()}

    out = {
        "dataset": args.dataset,
        "lambda_grid": lambda_curve,
        "per_class_recall_by_lambda": per_class_recall_by_lambda,
        "selected_lambda": best_lambda,
        "tm_argmax": report(tm_only_preds, "TM argmax"),
        "stabilis_only": report(stab_only_preds, "STABILIS only"),
        "fused": report(fused_preds, f"Fused (lambda={best_lambda})"),
        "n_eval": int(len(eval_idx)),
    }
    with open(f"{base}/fusion_results_seed{args.seed}_thr{args.pi_thr}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved fusion results.")


if __name__ == "__main__":
    main()
