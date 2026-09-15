"""
Binary Clause-Activation Signatures (CAS): normal-vs-attack fallback arm.

The per-class CAS question's fallback: collapse all attack families into a
single ATTACK class, train a dedicated 2-class TM (normal vs attack) on the
booleanized data, and extract the STABILIS/CPSS signature of the ATTACK class
(inculpatory = evidence FOR attack; exculpatory = evidence AGAINST, i.e.
stably normal-indicating clauses), with the same E[V] certificate machinery
as the per-class pipeline (stabilis.py, FAMILIES patched to the binary pair).

Evaluates, on a held-out eval split disjoint from the CPSS validation split:
  - CAS signature-match attribution (argmax of normal/attack signature scores)
  - the binary TM's own argmax baseline
  - AUROC/AUPRC of both the CAS attack score and the TM attack margin

Also decodes the attack signature's clauses into human-readable literal
conjunctions (the forensic evidence list), reusing decode_signatures.py's
clause decoder.

Usage:
  python3 scripts/binary_cas.py {iotm,medsec,wustl} --seed 42 [--pi_thr 0.6 --B 15]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from tmu.models.classification.vanilla_classifier import TMClassifier

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod
from decode_signatures import decode_clause

CONFIG = {
    # families[0] MUST be the benign/normal class (label index 0 in the npz).
    "iotm":  {"dir": "results",       "families": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
              "clauses": 400, "T": 320, "s": 8.0, "epochs": 25},
    "medsec": {"dir": "results_medsec", "families": ["Benign", "Reconnaissance", "Initial access",
                                                     "Lateral movement", "Exfiltration"],
               "clauses": 400, "T": 320, "s": 8.0, "epochs": 25},
    "wustl": {"dir": "results_wustl", "families": ["normal", "Spoofing", "Data Alteration"],
              "clauses": 200, "T": 160, "s": 8.0, "epochs": 25},
}
BINARY_FAMILIES = ["normal", "attack"]


def train_binary_tm(Xtr, ytr_bin, cfg, seed, model_path):
    tm = TMClassifier(number_of_clauses=cfg["clauses"], T=cfg["T"], s=cfg["s"],
                      weighted_clauses=True, platform="CPU", seed=seed)
    t0 = time.time()
    for _ in range(cfg["epochs"]):
        tm.fit(Xtr, ytr_bin)
    print(f"binary TM trained in {time.time()-t0:.1f}s on {Xtr.shape[0]} rows "
          f"(attack prevalence {ytr_bin.mean():.3f})", flush=True)
    with open(model_path, "wb") as f:
        pickle.dump(tm, f)
    return tm


def binary_metrics(y_true, pred, score):
    rep = classification_report(y_true, pred, target_names=BINARY_FAMILIES, output_dict=True)
    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "attack_precision": float(rep["attack"]["precision"]),
        "attack_recall": float(rep["attack"]["recall"]),
        "attack_f1": float(rep["attack"]["f1-score"]),
        "normal_recall": float(rep["normal"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }
    return out


def decode_binary_signature(tm, sig_entry, diag_fam, literal_names, n_clauses, top_k=None):
    """Decode one binary family's signature clauses to literal conjunctions."""
    rows = {"inculpatory": [], "exculpatory": []}
    for role, S in [("inculpatory", sig_entry["pos"]), ("exculpatory", sig_entry["neg"])]:
        for j in S:
            src_class = j // n_clauses
            clause_within = j % n_clauses
            polarity, lits = decode_clause(tm, src_class, clause_within, literal_names)
            rows[role].append({
                "clause_id": int(j),
                "source_bank": BINARY_FAMILIES[src_class],
                "polarity": polarity,
                "pi_hat": round(float(diag_fam["pi_hat"][j]), 3),
                "mean_coef": round(float(diag_fam["mean_coef"][j]), 4),
                "n_literals": len(lits),
                "literals": lits,
            })
        rows[role].sort(key=lambda r: -r["pi_hat"])
    if top_k:
        rows = {k: v[:top_k] for k, v in rows.items()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(CONFIG.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi_thr", type=float, default=0.6)
    ap.add_argument("--B", type=int, default=15)
    args = ap.parse_args()

    cfg = CONFIG[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{cfg['dir']}"
    tag = f"{args.dataset}_seed{args.seed}"

    d = np.load(f"{base}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    ytr_bin = (ytr != 0).astype(np.uint32)  # families[0] is benign/normal
    yte_bin = (yte != 0).astype(np.uint32)
    print(f"[{tag}] test rows {len(yte_bin)}, attack prevalence {yte_bin.mean():.3f}", flush=True)

    model_path = f"{base}/binary_tm_model_seed{args.seed}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            tm = pickle.load(f)
        print(f"loaded cached {model_path}", flush=True)
    else:
        tm = train_binary_tm(Xtr, ytr_bin, cfg, args.seed, model_path)

    print("extracting activations Z = tm.transform(Xte) ...", flush=True)
    t0 = time.time()
    Z = tm.transform(Xte).astype(np.float64)
    print(f"Z shape {Z.shape} ({time.time()-t0:.1f}s)", flush=True)

    # same val/eval convention as the per-class pipeline (disjoint halves)
    rng = np.random.RandomState(0)
    n = Z.shape[0]
    perm = rng.permutation(n)
    val_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    stab_mod.FAMILIES = BINARY_FAMILIES
    signatures, diagnostics = stab_mod.build_signatures(
        Z[val_idx], yte_bin[val_idx], pi_thr=args.pi_thr, B=args.B, seed=args.seed)

    # --- eval: CAS attribution vs TM argmax ---
    Z_eval, y_eval, X_eval = Z[eval_idx], yte_bin[eval_idx], Xte[eval_idx]
    scores = stab_mod.attribution_scores(Z_eval, signatures, diagnostics)
    cas_pred = np.argmax(scores, axis=1)
    cas_metrics = binary_metrics(y_eval, cas_pred, scores[:, 1])

    tm_pred, class_sums = tm.predict(X_eval, return_class_sums=True)
    class_sums = np.array(class_sums, dtype=np.float64)
    tm_margin = np.clip(class_sums[:, 1], 0, cfg["T"]) / cfg["T"]
    tm_metrics = binary_metrics(y_eval, tm_pred, tm_margin)

    print(f"\n[{tag}] TM-argmax baseline : attack F1={tm_metrics['attack_f1']:.4f} "
          f"R={tm_metrics['attack_recall']:.4f} P={tm_metrics['attack_precision']:.4f} "
          f"AUROC={tm_metrics['auroc']:.4f}", flush=True)
    print(f"[{tag}] CAS attribution    : attack F1={cas_metrics['attack_f1']:.4f} "
          f"R={cas_metrics['attack_recall']:.4f} P={cas_metrics['attack_precision']:.4f} "
          f"AUROC={cas_metrics['auroc']:.4f}", flush=True)

    # --- decode attack signature to literals ---
    with open(f"{base}/literal_names.json") as f:
        literal_names = json.load(f)["literal_names"]
    n_clauses = tm.number_of_clauses
    decoded = {
        fam: decode_binary_signature(tm, signatures[fam], diagnostics[fam], literal_names, n_clauses)
        for fam in BINARY_FAMILIES
    }

    out = {
        "dataset": args.dataset, "seed": args.seed, "pi_thr": args.pi_thr, "B": args.B,
        "tm_config": {k: cfg[k] for k in ("clauses", "T", "s", "epochs")},
        "n_test": int(n), "attack_prevalence_test": float(yte_bin.mean()),
        "signatures": signatures,
        "diagnostics": diagnostics,  # full pi_hat/mean_coef: lets figure
                                     # scripts score without re-running CPSS
        "signature_summary": {
            fam: {k: diagnostics[fam][k] for k in
                  ("signature_size", "signature_size_signed", "signature_size_exculpatory",
                   "q_lambda_estimate", "E_V_bound")}
            for fam in BINARY_FAMILIES},
        "cas_attribution": cas_metrics,
        "tm_argmax": tm_metrics,
        "n_eval": int(len(eval_idx)),
        "val_idx": val_idx.tolist(), "eval_idx": eval_idx.tolist(),
    }
    res_path = f"{base}/binary_cas_results_seed{args.seed}_thr{args.pi_thr}.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    dec_path = f"{base}/binary_cas_decoded_seed{args.seed}_thr{args.pi_thr}.json"
    with open(dec_path, "w") as f:
        json.dump(decoded, f, indent=2)
    print(f"saved {res_path}\nsaved {dec_path}", flush=True)


if __name__ == "__main__":
    main()
