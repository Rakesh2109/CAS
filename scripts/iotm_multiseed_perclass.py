"""
Multi-seed per-class CAS on IoTM (CICIoMT2024), closing the single-seed hole
in the per-class result and searching for the best verified F1.

Per seed (42, 7, 123), on the 10-bin booleanized data:
  1. train a weighted multiclass TM (400 clauses/class, T=320, s=8.0, 25 epochs)
  2. CPSS signatures per family (B=15, seeded liblinear -> reproducible)
  3. Task A: TM-argmax vs CAS attribution macro-F1 on held-out eval half
  4. pi_thr sweep {0.5..0.9} from the SAME pi_hat (free re-thresholding):
     macro-F1 vs mean signature size -- the "compact without losing F1" curve
  5. fusion: tm_score + lambda * cas_score, lambda picked on the validation
     half (no leakage), evaluated once on the eval half

Outputs results/iotm_multiseed_perclass.json + per-seed signature/model files
(suffixed _nbins10 to keep them distinct from the binary arm's 5-bin artifacts).
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod
from tmu.models.classification.vanilla_classifier import TMClassifier

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
CLAUSES, T, S, EPOCHS = 400, 320, 8.0, 25
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
stab_mod.FAMILIES = FAMILIES


def signatures_at_thr(pi_hat, mean_coef, thr):
    S_pos = np.where((pi_hat >= thr) & (mean_coef > 0))[0]
    S_neg = np.where((pi_hat >= thr) & (mean_coef < 0))[0]
    return S_pos, S_neg


def cas_scores(Z, pi_hats, mean_coefs, thr):
    """attribution scores for all families at threshold thr (reuses stab logic)."""
    n = Z.shape[0]
    scores = np.zeros((n, len(FAMILIES)))
    for i, fam in enumerate(FAMILIES):
        S_pos, S_neg = signatures_at_thr(pi_hats[fam], mean_coefs[fam], thr)
        pi_hat = pi_hats[fam]
        denom = pi_hat[S_pos].sum() if len(S_pos) else 0.0
        if denom <= 0:
            continue
        pos_term = Z[:, S_pos] @ pi_hat[S_pos] if len(S_pos) else 0.0
        neg_term = Z[:, S_neg] @ pi_hat[S_neg] if len(S_neg) else 0.0
        scores[:, i] = (pos_term - neg_term) / denom
    return scores


def run_seed(seed, Xtr, ytr, Xte, yte, B):
    print(f"\n===== seed {seed} =====", flush=True)
    model_path = f"{BASE}/tm_model_nbins10_seed{seed}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            tm = pickle.load(f)
        print("loaded cached model", flush=True)
    else:
        tm = TMClassifier(number_of_clauses=CLAUSES, T=T, s=S,
                          weighted_clauses=True, platform="CPU", seed=seed)
        t0 = time.time()
        for _ in range(EPOCHS):
            tm.fit(Xtr, ytr)
        print(f"trained in {time.time()-t0:.0f}s", flush=True)
        with open(model_path, "wb") as f:
            pickle.dump(tm, f)

    Z = tm.transform(Xte).astype(np.float64)
    rng = np.random.RandomState(0)
    n = Z.shape[0]
    perm = rng.permutation(n)
    val_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    # CPSS once per family at the primary threshold; pi_hat reused for the sweep
    pi_hats, mean_coefs, bounds = {}, {}, {}
    for cls_idx, fam in enumerate(FAMILIES):
        y_bin = (yte[val_idx] == cls_idx).astype(int)
        pi_hat, mean_coef = stab_mod.cpss_proportions(Z[val_idx], y_bin, B=B, seed=seed + cls_idx)
        pi_hats[fam], mean_coefs[fam] = pi_hat, mean_coef
        q_est = float(np.mean(pi_hat) * len(pi_hat))
        bounds[fam] = stab_mod.error_bound(pi_hat, 0.6, q_est)
        print(f"  [{fam}] CPSS done", flush=True)

    tm_pred, class_sums = tm.predict(Xte, return_class_sums=True)
    class_sums = np.array(class_sums, dtype=np.float64)
    tm_score = np.clip(class_sums, 0, T) / T
    tm_f1_eval = f1_score(yte[eval_idx], tm_pred[eval_idx], average="macro")

    out = {"seed": seed, "tm_argmax_macro_f1": float(tm_f1_eval),
           "E_V_bounds_thr0.6": bounds, "pi_thr_sweep": {}, "fusion": {}}

    # --- pi_thr sweep (free re-thresholding of the same pi_hat) ---
    for thr in PI_THRS:
        cas_eval = cas_scores(Z[eval_idx], pi_hats, mean_coefs, thr)
        f1 = f1_score(yte[eval_idx], np.argmax(cas_eval, axis=1), average="macro")
        sizes = [int(len(signatures_at_thr(pi_hats[f], mean_coefs[f], thr)[0])) for f in FAMILIES]
        out["pi_thr_sweep"][str(thr)] = {
            "cas_macro_f1": float(f1), "signature_sizes_pos": sizes,
            "mean_sig_size": float(np.mean(sizes))}
        print(f"  thr={thr}: CAS macro-F1={f1:.4f} mean|S+|={np.mean(sizes):.0f}", flush=True)

    # --- fusion at thr=0.6: lambda tuned on val, evaluated once on eval ---
    cas_val = cas_scores(Z[val_idx], pi_hats, mean_coefs, 0.6)
    cas_eval = cas_scores(Z[eval_idx], pi_hats, mean_coefs, 0.6)
    best = None
    for lam in LAMBDAS:
        fused_val = tm_score[val_idx] + lam * cas_val
        f1 = f1_score(yte[val_idx], np.argmax(fused_val, axis=1), average="macro")
        if best is None or f1 > best[1]:
            best = (lam, f1)
    lam = best[0]
    fused_eval = tm_score[eval_idx] + lam * cas_eval
    fusion_f1 = f1_score(yte[eval_idx], np.argmax(fused_eval, axis=1), average="macro")
    out["fusion"] = {"lambda": lam, "val_macro_f1": float(best[1]),
                     "eval_macro_f1": float(fusion_f1)}
    print(f"  fusion: lambda*={lam} eval macro-F1={fusion_f1:.4f} (TM {tm_f1_eval:.4f})", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=15, help="CPSS subsamples (15 or 25)")
    args = ap.parse_args()
    B = args.B

    d = np.load(f"{BASE}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    print(f"data: Xtr {Xtr.shape}, Xte {Xte.shape}, B={B}", flush=True)

    runs = [run_seed(s, Xtr, ytr, Xte, yte, B) for s in SEEDS]

    summary = {"seeds": SEEDS, "B": B, "runs": runs,
               "tm_argmax_macro_f1_mean": float(np.mean([r["tm_argmax_macro_f1"] for r in runs])),
               "tm_argmax_macro_f1_std": float(np.std([r["tm_argmax_macro_f1"] for r in runs]))}
    for thr in PI_THRS:
        vals = [r["pi_thr_sweep"][str(thr)]["cas_macro_f1"] for r in runs]
        summary[f"cas_thr{thr}_macro_f1_mean"] = float(np.mean(vals))
        summary[f"cas_thr{thr}_macro_f1_std"] = float(np.std(vals))
    fus = [r["fusion"]["eval_macro_f1"] for r in runs]
    summary["fusion_macro_f1_mean"] = float(np.mean(fus))
    summary["fusion_macro_f1_std"] = float(np.std(fus))

    out_path = f"{BASE}/iotm_multiseed_perclass_B{B}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
