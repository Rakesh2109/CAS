"""
Validation of the sign-filtering fix (paper: Method sign filtering,
Results "Validation of the construction").

For a trained TM seed, loads the *saved* CPSS signature artifacts (which
store threshold-independent pi_hat and mean_coef plus the exact eval split)
and scores held-out attribution three ways:

  1. TM-argmax baseline (reference)
  2. sign-agnostic (the bug): s_l(x) = sum_{j in S_l} pi_hat_j z_j(x)
     over the full sign-agnostic selected support S_l = S+ u S-
  3. signed (paper Eq. 1): inculpatory minus exculpatory, normalized by
     the inculpatory weight

Writes results/sign_fix_validation.json. No CPSS refit is needed: pi_hat
and mean_coef stored in stabilis_signatures_seed*_thr*.json are
threshold-independent, and pos/neg membership at the file's own threshold
is stored.
"""
import json
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from stabilis import FAMILIES, attribution_scores

BASE = "/FPTM/CAS_IoMT_Empirical/results"


def reconstruct_signed(signatures, diagnostics):
    """Saved artifacts store signatures[fam] as the flat sign-agnostic list
    S_l; recover the pos/neg split from the stored mean signed coefficient."""
    signed = {}
    for fam in FAMILIES:
        v = signatures[fam]
        if isinstance(v, dict):
            signed[fam] = v
            continue
        mean_coef = np.array(diagnostics[fam]["mean_coef"])
        signed[fam] = {
            "pos": [j for j in v if mean_coef[j] > 0],
            "neg": [j for j in v if mean_coef[j] < 0],
        }
    return signed


def sign_agnostic_scores(Z, signatures, diagnostics):
    """The pre-fix scorer: naive pi-weighted sum over the full sign-agnostic
    selected support (S+ union S-), no sign split, no normalization."""
    n = Z.shape[0]
    scores = np.zeros((n, len(FAMILIES)))
    for i, fam in enumerate(FAMILIES):
        v = signatures[fam]
        S = (list(v["pos"]) + list(v["neg"])) if isinstance(v, dict) else list(v)
        pi_hat = np.array(diagnostics[fam]["pi_hat"])
        if len(S):
            scores[:, i] = Z[:, S] @ pi_hat[S]
    return scores


def run(seed, pi_thr):
    with open(f"{BASE}/tm_model_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    with open(f"{BASE}/stabilis_signatures_seed{seed}_thr{pi_thr}.json") as f:
        sig_data = json.load(f)
    signatures, diagnostics = sig_data["signatures"], sig_data["diagnostics"]
    eval_idx = np.array(sig_data["eval_idx"])

    t0 = time.time()
    Z_eval = tm.transform(Xte[eval_idx]).astype(np.float64)
    y_eval = yte[eval_idx]

    tm_f1 = f1_score(y_eval, tm.predict(Xte[eval_idx]), average="macro")
    naive_f1 = f1_score(
        y_eval, np.argmax(sign_agnostic_scores(Z_eval, signatures, diagnostics), axis=1),
        average="macro")
    signed = reconstruct_signed(signatures, diagnostics)
    signed_f1 = f1_score(
        y_eval, np.argmax(attribution_scores(Z_eval, signed, diagnostics), axis=1),
        average="macro")

    return {
        "seed": seed, "pi_thr": pi_thr, "B": sig_data.get("B"),
        "n_eval": int(len(eval_idx)),
        "tm_argmax_macro_f1": float(tm_f1),
        "sign_agnostic_macro_f1": float(naive_f1),
        "signed_macro_f1": float(signed_f1),
        "transform_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    # default: seed 42 at the two thresholds discussed in the paper
    pairs = [(42, 0.6), (42, 0.8)]
    if len(sys.argv) > 1:
        pairs = []
        for a in sys.argv[1:]:
            s, t = a.split(":")
            pairs.append((int(s), float(t)))
    out = []
    for seed, thr in pairs:
        r = run(seed, thr)
        print(json.dumps(r, indent=2))
        out.append(r)
    path = f"{BASE}/sign_fix_validation.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")
