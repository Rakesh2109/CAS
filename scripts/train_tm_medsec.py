import argparse
import json
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tmu.models.classification.vanilla_classifier import TMClassifier

FAMILIES = ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"]
OUT_DIR = "/FPTM/CAS_IoMT_Empirical/results_medsec"


def load_data():
    d = np.load(f"{OUT_DIR}/booleanized_data.npz")
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=int, default=400)
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_data()
    print(f"training on {Xtr.shape}, testing on {Xte.shape}")

    tm = TMClassifier(number_of_clauses=args.clauses, T=args.T, s=args.s,
                       weighted_clauses=True, platform="CPU", seed=args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        ep_t0 = time.time()
        tm.fit(Xtr, ytr)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            preds = tm.predict(Xte)
            f1 = f1_score(yte, preds, average="macro")
            print(f"epoch {ep+1}/{args.epochs} time={time.time()-ep_t0:.1f}s macro-F1={f1:.4f}")
    total_time = time.time() - t0

    preds = tm.predict(Xte)
    macro_f1 = f1_score(yte, preds, average="macro")
    report = classification_report(yte, preds, target_names=FAMILIES, output_dict=True)
    cm = confusion_matrix(yte, preds, labels=list(range(len(FAMILIES))))
    print(f"\nFINAL macro-F1={macro_f1:.4f} total_train_time={total_time:.1f}s")
    print(classification_report(yte, preds, target_names=FAMILIES))

    results = {
        "macro_f1": macro_f1, "per_class_report": report, "confusion_matrix": cm.tolist(),
        "hyperparams": vars(args), "train_time_s": total_time,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
    }
    with open(f"{OUT_DIR}/tm_argmax_baseline_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    import pickle
    with open(f"{OUT_DIR}/tm_model_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(tm, f)
    print("saved model.")


if __name__ == "__main__":
    main()
