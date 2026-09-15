"""
Run STABILIS signature extraction + Task A attribution eval for any of the
dataset dirs (IoTM/medsec/wustl), reusing stabilis.py's CPSS + attribution
logic with FAMILIES swapped per dataset (same monkey-patch pattern already
used in task_b_lofo.py).
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
    "medsec": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
    "wustl": ["normal", "Spoofing", "Data Alteration"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["medsec", "wustl"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi_thr", type=float, default=0.6)
    ap.add_argument("--B", type=int, default=15)
    args = ap.parse_args()

    base = f"/FPTM/CAS_IoMT_Empirical/results_{args.dataset}"
    families = DATASET_FAMILIES[args.dataset]
    stab_mod.FAMILIES = families

    with open(f"{base}/tm_model_seed{args.seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{base}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    print("extracting activations...")
    t0 = time.time()
    Z = tm.transform(Xte).astype(np.float64)
    print(f"Z shape {Z.shape} ({time.time()-t0:.1f}s)")

    rng = np.random.RandomState(0)
    n = Z.shape[0]
    perm = rng.permutation(n)
    val_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    signatures, diagnostics = stab_mod.build_signatures(Z[val_idx], yte[val_idx], pi_thr=args.pi_thr, B=args.B, seed=args.seed)

    with open(f"{base}/stabilis_signatures_seed{args.seed}_thr{args.pi_thr}.json", "w") as f:
        json.dump({"signatures": signatures, "diagnostics": diagnostics, "pi_thr": args.pi_thr,
                    "B": args.B, "seed": args.seed, "val_idx": val_idx.tolist(), "eval_idx": eval_idx.tolist()}, f)

    Z_eval, y_eval, X_eval = Z[eval_idx], yte[eval_idx], Xte[eval_idx]

    tm_preds = tm.predict(X_eval)
    tm_f1 = f1_score(y_eval, tm_preds, average="macro")
    tm_report = classification_report(y_eval, tm_preds, target_names=families, output_dict=True)
    tm_cm = confusion_matrix(y_eval, tm_preds, labels=list(range(len(families))))

    scores = stab_mod.attribution_scores(Z_eval, signatures, diagnostics)
    stab_preds = np.argmax(scores, axis=1)
    stab_f1 = f1_score(y_eval, stab_preds, average="macro")
    stab_report = classification_report(y_eval, stab_preds, target_names=families, output_dict=True)
    stab_cm = confusion_matrix(y_eval, stab_preds, labels=list(range(len(families))))

    print(f"\nTM-argmax baseline   macro-F1 = {tm_f1:.4f}")
    print(f"STABILIS attribution macro-F1 = {stab_f1:.4f}")
    print(classification_report(y_eval, tm_preds, target_names=families))
    print(classification_report(y_eval, stab_preds, target_names=families))

    out = {
        "dataset": args.dataset, "seed": args.seed, "pi_thr": args.pi_thr,
        "tm_argmax": {"macro_f1": tm_f1, "report": tm_report, "confusion_matrix": tm_cm.tolist()},
        "stabilis": {"macro_f1": stab_f1, "report": stab_report, "confusion_matrix": stab_cm.tolist()},
        "n_eval": int(len(eval_idx)),
    }
    with open(f"{base}/task_a_results_seed{args.seed}_thr{args.pi_thr}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved.")


if __name__ == "__main__":
    main()
