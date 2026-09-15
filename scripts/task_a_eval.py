"""
Task A -- closed-set attribution eval: STABILIS signature-score attribution
vs. direct TM-argmax baseline (forensic_fingerprint_methods.md Sec 0.4).
"""
import json
import pickle
import sys
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from stabilis import FAMILIES, attribution_scores


def main(seed=42, pi_thr=0.6):
    base = "/FPTM/CAS_IoMT_Empirical/results"
    with open(f"{base}/tm_model_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{base}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    with open(f"{base}/stabilis_signatures_seed{seed}_thr{pi_thr}.json") as f:
        sig_data = json.load(f)
    signatures, diagnostics = sig_data["signatures"], sig_data["diagnostics"]
    eval_idx = np.array(sig_data["eval_idx"])

    Z = tm.transform(Xte).astype(np.float64)
    Z_eval, y_eval = Z[eval_idx], yte[eval_idx]
    X_eval = Xte[eval_idx]

    # baseline: direct TM argmax
    tm_preds = tm.predict(X_eval)
    tm_f1 = f1_score(y_eval, tm_preds, average="macro")
    tm_report = classification_report(y_eval, tm_preds, target_names=FAMILIES, output_dict=True)
    tm_cm = confusion_matrix(y_eval, tm_preds, labels=list(range(len(FAMILIES))))

    # STABILIS: attribution via signature match score
    scores = attribution_scores(Z_eval, signatures, diagnostics)
    stab_preds = np.argmax(scores, axis=1)
    stab_f1 = f1_score(y_eval, stab_preds, average="macro")
    stab_report = classification_report(y_eval, stab_preds, target_names=FAMILIES, output_dict=True)
    stab_cm = confusion_matrix(y_eval, stab_preds, labels=list(range(len(FAMILIES))))

    # runner-up margin as confidence signal
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    print(f"TM-argmax baseline   macro-F1 = {tm_f1:.4f}")
    print(f"STABILIS attribution macro-F1 = {stab_f1:.4f}")
    print("\n--- TM-argmax ---")
    print(classification_report(y_eval, tm_preds, target_names=FAMILIES))
    print("\n--- STABILIS ---")
    print(classification_report(y_eval, stab_preds, target_names=FAMILIES))

    out = {
        "seed": seed, "pi_thr": pi_thr,
        "tm_argmax": {"macro_f1": tm_f1, "report": tm_report, "confusion_matrix": tm_cm.tolist()},
        "stabilis": {"macro_f1": stab_f1, "report": stab_report, "confusion_matrix": stab_cm.tolist(),
                      "mean_margin": float(margin.mean())},
        "n_eval": int(len(eval_idx)),
    }
    with open(f"{base}/task_a_results_seed{seed}_thr{pi_thr}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved task A results.")
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    pi_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    main(seed, pi_thr)
