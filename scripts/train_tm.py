"""
Train a multiclass Tsetlin Machine (TMU TMClassifier) on the booleanized
CICIoMT2024 family-level data. Sanity gate: macro-F1 >= 0.85 closed-set
(forensic_fingerprint_methods.md Sec 0.1 point 3).
"""
import argparse
import json
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tmu.models.classification.vanilla_classifier import TMClassifier

FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def load_data(path="/FPTM/CAS_IoMT_Empirical/results/booleanized_data.npz"):
    d = np.load(path)
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"]


def train(Xtr, ytr, Xte, yte, number_of_clauses=250, T=200, s=5.0, epochs=20, seed=42, weighted_clauses=True):
    tm = TMClassifier(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        weighted_clauses=weighted_clauses,
        platform="CPU",
        seed=seed,
    )
    t0 = time.time()
    for ep in range(epochs):
        ep_t0 = time.time()
        tm.fit(Xtr, ytr)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            preds = tm.predict(Xte)
            macro_f1 = f1_score(yte, preds, average="macro")
            print(f"epoch {ep+1}/{epochs}  time={time.time()-ep_t0:.1f}s  test macro-F1={macro_f1:.4f}")
    total_time = time.time() - t0
    return tm, total_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=int, default=250)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="/FPTM/CAS_IoMT_Empirical/results/tm_model.npz")
    ap.add_argument("--quick", action="store_true", help="subsample for a fast timing test")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_data()
    if args.quick:
        rng = np.random.RandomState(0)
        idx_tr = rng.choice(len(Xtr), size=5000, replace=False)
        idx_te = rng.choice(len(Xte), size=2000, replace=False)
        Xtr, ytr = Xtr[idx_tr], ytr[idx_tr]
        Xte, yte = Xte[idx_te], yte[idx_te]

    print(f"training on {Xtr.shape}, testing on {Xte.shape}")
    tm, total_time = train(Xtr, ytr, Xte, yte, args.clauses, args.T, args.s, args.epochs, args.seed)

    preds, class_sums = tm.predict(Xte, return_class_sums=True)
    macro_f1 = f1_score(yte, preds, average="macro")
    report = classification_report(yte, preds, target_names=FAMILIES, output_dict=True)
    cm = confusion_matrix(yte, preds, labels=list(range(len(FAMILIES))))

    print(f"\nFINAL macro-F1={macro_f1:.4f}  total_train_time={total_time:.1f}s")
    print(classification_report(yte, preds, target_names=FAMILIES))

    results = {
        "macro_f1": macro_f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "hyperparams": {"clauses": args.clauses, "T": args.T, "s": args.s, "epochs": args.epochs, "seed": args.seed},
        "train_time_s": total_time,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
    }
    tag = "quick" if args.quick else f"seed{args.seed}"
    with open(f"/FPTM/CAS_IoMT_Empirical/results/tm_argmax_baseline_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    if not args.quick:
        import pickle
        with open(args.out.replace(".npz", f"_seed{args.seed}.pkl"), "wb") as f:
            pickle.dump(tm, f)
        print("saved model to", args.out.replace(".npz", f"_seed{args.seed}.pkl"))


if __name__ == "__main__":
    main()
