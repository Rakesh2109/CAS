"""
Method 2 -- SIMPLEXPRO: compositional prototype signatures in Aitchison
geometry (forensic_fingerprint_methods.md Sec "METHOD 2"). Reuses the
already-trained TM models from each dataset (no retraining needed):

  h(x)   = (w_pos * z(x)) / sum(w_pos * z(x))          in the simplex Delta^{m-1}
  w_pos  = max(w, eps)   -- mandatory correctness fix: TMU initializes/updates
                            clause weights signed, and negative components
                            break the simplex/clr construction.
  clr(h)_j = log(h_j / g(h)), g(h) = geometric mean of h
  mu_l   = mean_i clr(h_i)   over family l's validation samples  (prototype)
  pred(x) = argmin_l || clr(h(x)) - mu_l ||             (Aitchison distance,
                                                           clr coords == ilr
                                                           isometry up to
                                                           rotation)

Zero-handling: empty firing (sum(w_pos*z)==0) falls back to the uniform
composition (every part = 1/m); zeros in h are handled via the standard
"simple replacement" multiplicative rule (Martin-Fernandez et al. 2003) with
pseudo-count delta, which preserves closure-to-1 and ratios among the
nonzero parts (not naive delta-then-renormalize).
"""
import argparse
import json
import pickle
import sys
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

DATASET_FAMILIES = {
    "iotm": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
    "medsec": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
    "wustl": ["normal", "Spoofing", "Data Alteration"],
}
DATASET_DIRS = {"iotm": "results", "medsec": "results_medsec", "wustl": "results_wustl"}
EPS_WEIGHT = 1.0  # w_pos = max(w, eps); TMU weights are integers, min useful positive weight is 1


def full_clause_weight_vector(tm):
    """Concatenate weight_banks[c].get_weights() across classes, in the same
    column order transform() uses (class0's clauses, then class1's, ...)."""
    parts = []
    for c in tm.weight_banks.classes():
        parts.append(np.array(tm.weight_banks[c].get_weights(), dtype=np.float64))
    return np.concatenate(parts)


def compositional_profile(Z, w_pos, delta):
    """Z: (n, m) binary clause activations. Returns h: (n, m) simplex points."""
    n, m = Z.shape
    num = Z * w_pos[None, :]
    denom = num.sum(axis=1, keepdims=True)
    empty = (denom[:, 0] == 0)
    h = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
    if empty.any():
        h[empty] = 1.0 / m  # empty-firing fallback: uniform composition

    # simple replacement strategy (Martin-Fernandez et al. 2003): preserves
    # closure to 1 and ratios among nonzero parts, unlike naive delta+renorm.
    zero_mask = (h == 0)
    n_zero = zero_mask.sum(axis=1, keepdims=True)
    scale = 1.0 - n_zero * delta
    scale = np.clip(scale, 1e-6, None)
    h_rep = np.where(zero_mask, delta, h * scale)
    return h_rep


def clr(h):
    log_h = np.log(h)
    g = log_h.mean(axis=1, keepdims=True)
    return log_h - g


def build_prototypes(H_clr, y, families):
    prototypes = {}
    for i, fam in enumerate(families):
        mask = y == i
        prototypes[fam] = H_clr[mask].mean(axis=0) if mask.sum() else np.zeros(H_clr.shape[1])
    return prototypes


def attribution_predict(H_clr, prototypes, families):
    n = H_clr.shape[0]
    dists = np.zeros((n, len(families)))
    for i, fam in enumerate(families):
        dists[:, i] = np.linalg.norm(H_clr - prototypes[fam][None, :], axis=1)
    return dists.argmin(axis=1), dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASET_FAMILIES.keys()))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    families = DATASET_FAMILIES[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{DATASET_DIRS[args.dataset]}"

    with open(f"{base}/tm_model_seed{args.seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{base}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    print("computing Z, weight vector...")
    Z = tm.transform(Xte).astype(np.float64)
    w = full_clause_weight_vector(tm)
    assert w.shape[0] == Z.shape[1], f"weight vector {w.shape} vs Z cols {Z.shape[1]}"
    w_pos = np.maximum(w, EPS_WEIGHT)

    m, n = Z.shape[1], Z.shape[0]
    delta = 1.0 / (m * n)
    print(f"m={m} literals-worth-of-clauses, n={n}, delta={delta:.3e}")

    H = compositional_profile(Z, w_pos, delta)
    H_clr = clr(H)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    val_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    prototypes = build_prototypes(H_clr[val_idx], yte[val_idx], families)

    preds, dists = attribution_predict(H_clr[eval_idx], prototypes, families)
    y_eval = yte[eval_idx]
    macro_f1 = f1_score(y_eval, preds, average="macro")
    report = classification_report(y_eval, preds, target_names=families, output_dict=True)
    cm = confusion_matrix(y_eval, preds, labels=list(range(len(families))))

    # margin diagnostic: minimum inter-prototype distance (Delta from Thm 1's corollary)
    proto_mat = np.stack([prototypes[f] for f in families])
    inter_dists = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            inter_dists.append(np.linalg.norm(proto_mat[i] - proto_mat[j]))
    margin = float(min(inter_dists)) if inter_dists else None

    print(f"\nSIMPLEXPRO macro-F1 = {macro_f1:.4f}  (min inter-prototype margin Delta = {margin:.3f})")
    print(classification_report(y_eval, preds, target_names=families))

    with open(f"{base}/tm_argmax_baseline_seed{args.seed}.json") as f:
        tm_baseline = json.load(f)

    out = {
        "dataset": args.dataset, "seed": args.seed, "delta": delta, "m": int(m),
        "macro_f1": macro_f1, "report": report, "confusion_matrix": cm.tolist(),
        "margin_delta": margin, "n_eval": int(len(eval_idx)),
        "tm_argmax_macro_f1_reference": tm_baseline["macro_f1"],
    }
    with open(f"{base}/simplexpro_results_seed{args.seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved SIMPLEXPRO results.")


if __name__ == "__main__":
    main()
