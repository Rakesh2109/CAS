"""
Train one Tsetlin Machine on the FULL CICIoMT2024 official split.

  --arm multiclass  : 6-family, results/full_bool_nbins10.npz, macro-F1
  --arm binary      : normal-vs-attack, results/full_bool_nbins5.npz, F1/R/P/AUROC

TMU CPU is single-threaded; run several (arm, seed) instances in parallel.
Config defaults to the paper's C=400/T=320/s=8 (grid-search-verified within
~1pp of optimal on the 22-feature booleanization).

  python3 scripts/full_train.py --arm multiclass --seed 42
"""
import argparse
import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tmu.models.classification.vanilla_classifier import TMClassifier

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["multiclass", "binary"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clauses", type=int, default=400)
    ap.add_argument("--T", type=int, default=320)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--prefix", default="full")
    args = ap.parse_args()

    nb = 10 if args.arm == "multiclass" else 5
    d = np.load(f"{BASE}/{args.prefix}_bool_nbins{nb}.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    if args.arm == "binary":
        ytr = (ytr != 0)
        yte = (yte != 0)
    ytr = np.ascontiguousarray(ytr, dtype=np.uint32)
    yte = np.ascontiguousarray(yte, dtype=np.uint32)
    Xtr = np.ascontiguousarray(Xtr, dtype=np.uint32)
    Xte = np.ascontiguousarray(Xte, dtype=np.uint32)
    tag = f"{args.prefix}_{args.arm}_seed{args.seed}"
    print(f"[{tag}] train {Xtr.shape} test {Xte.shape} "
          f"C={args.clauses} T={args.T} s={args.s}", flush=True)

    tm = TMClassifier(number_of_clauses=args.clauses, T=args.T, s=args.s,
                      weighted_clauses=True, platform="CPU", seed=args.seed)
    curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        et = time.time()
        tm.fit(Xtr, ytr)
        if ep % 2 == 0 or ep == args.epochs:
            pred = tm.predict(Xte)
            mf1 = float(f1_score(yte, pred, average="macro"))
            curve.append([ep, round(mf1, 4)])
            print(f"[{tag}] ep{ep}/{args.epochs} {time.time()-et:.0f}s "
                  f"macro-F1={mf1:.4f}", flush=True)
    train_s = time.time() - t0

    pred, cs = tm.predict(Xte, return_class_sums=True)
    cs = np.asarray(cs, dtype=np.float64)
    res = {"arm": args.arm, "seed": args.seed,
           "hyperparams": {"clauses": args.clauses, "T": args.T, "s": args.s,
                           "epochs": args.epochs},
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "train_time_s": round(train_s, 1), "curve": curve}

    if args.arm == "multiclass":
        res["macro_f1"] = float(f1_score(yte, pred, average="macro"))
        res["accuracy"] = float((pred == yte).mean())
        res["per_class"] = classification_report(
            yte, pred, target_names=FAMILIES, output_dict=True, zero_division=0)
        res["confusion_matrix"] = confusion_matrix(
            yte, pred, labels=list(range(6))).tolist()
        print(f"[{tag}] FINAL macro-F1={res['macro_f1']:.4f}", flush=True)
    else:
        margin = np.clip(cs[:, 1] - cs[:, 0], -args.T, args.T) / (2 * args.T) + 0.5
        res["attack_f1"] = float(f1_score(yte, pred, pos_label=1))
        res["attack_recall"] = float(recall_score(yte, pred, pos_label=1))
        res["attack_precision"] = float(precision_score(yte, pred, pos_label=1))
        res["macro_f1"] = float(f1_score(yte, pred, average="macro"))
        res["auroc"] = float(roc_auc_score(yte, margin))
        res["auprc"] = float(average_precision_score(yte, margin))
        res["attack_prevalence"] = float(yte.mean())
        res["confusion_matrix"] = confusion_matrix(yte, pred).tolist()
        print(f"[{tag}] FINAL attackF1={res['attack_f1']:.4f} "
              f"R={res['attack_recall']:.3f} P={res['attack_precision']:.3f} "
              f"AUROC={res['auroc']:.4f}", flush=True)

    with open(f"{BASE}/{tag}_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(f"{BASE}/tm_{tag}.pkl", "wb") as f:
        pickle.dump(tm, f)
    print(f"[{tag}] saved model + metrics ({train_s:.0f}s train)", flush=True)


if __name__ == "__main__":
    main()
