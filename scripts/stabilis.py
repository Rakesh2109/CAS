"""
Method 1 -- STABILIS: stability-selected clause signatures via
complementary-pairs stability selection (CPSS, Shah & Samworth 2013)
on l1-regularized family-vs-rest logistic models over clause activations.

Implements forensic_fingerprint_methods.md Sec "METHOD 1":
  - CPSS proportions pi_hat_j (union-over-lambda-grid convention)
  - Signature S_l = {j : pi_hat_j >= pi_thr}
  - Shah&Samworth E[V] bound as the reported certificate
  - Sign split of the signature by mean coefficient sign into inculpatory
    (S+) and exculpatory (S-) sets
  - Attribution score s_l(x) = [sum_{S+} pi_hat_j z_j(x)
    - sum_{S-} pi_hat_j z_j(x)] / sum_{S+} pi_hat_j
    (see attribution_scores(); the sign-agnostic naive sum is wrong --
    see validate_sign_fix.py)
"""
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def cpss_proportions(Z, y_bin, B=25, lambda_grid_C=None, seed=0, max_iter=200):
    """Complementary-pairs stability selection for one family-vs-rest split.

    Z: (n, m) binary clause activations. y_bin: (n,) in {0,1}.
    Returns (pi_hat, mean_coef): pi_hat (m,) is the union-over-grid selection
    proportion (sign-agnostic, used for the Shah-Samworth error-control
    certificate exactly as specified); mean_coef (m,) is the coefficient
    averaged over all (half, grid-point) fits, signed. A clause with
    pi_hat_j >= thr but mean_coef_j < 0 is "stably selected" but stably
    *negatively* associated with the family (evidence against it, e.g. it
    belongs to a different family's clause bank) -- attribution_scores()
    below must filter on sign, or the "evidence list" silently mixes for-
    and against- evidence and the argmax score becomes meaningless.
    """
    if lambda_grid_C is None:
        # C = 1/lambda; sweep within the sparsity-inducing region only (empirically
        # calibrated on this clause pool: C>~0.3 stops inducing real sparsity and
        # balloons the union-over-grid support, making the E[V] bound vacuous).
        lambda_grid_C = np.geomspace(0.001, 0.1, 6)

    n, m = Z.shape
    rng = np.random.RandomState(seed)
    selected_count = np.zeros(m, dtype=np.int64)
    coef_sum = np.zeros(m, dtype=np.float64)
    half = n // 2
    n_grid = len(lambda_grid_C)

    for b in range(B):
        perm = rng.permutation(n)
        halves = [perm[:half], perm[half:2 * half]]
        for h_idx in halves:
            Zh, yh = Z[h_idx], y_bin[h_idx]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue  # degenerate half (can't fit logistic reliably)
            union_selected = np.zeros(m, dtype=bool)
            for C in lambda_grid_C:
                clf = LogisticRegression(
                    penalty="l1", solver="liblinear", C=C,
                    max_iter=max_iter, class_weight="balanced",
                    random_state=0,  # liblinear shuffles data with the global
                                     # np.random state if unset -> CPSS fits
                                     # were not exactly reproducible otherwise
                )
                clf.fit(Zh, yh)
                coef = clf.coef_.ravel()
                union_selected |= (np.abs(coef) > 1e-8)
                coef_sum += coef
            selected_count += union_selected.astype(np.int64)

    pi_hat = selected_count / (2 * B)
    mean_coef = coef_sum / (2 * B * n_grid)
    return pi_hat, mean_coef


def error_bound(pi_hat, pi_thr, q_lambda_estimate):
    """Shah & Samworth (2013) unconditional bound on E|S ∩ L_theta| shape,
    reported here in the common operational E[V] <= q^2 / ((2*pi_thr-1)*p) form
    (Meinshausen-Buhlmann corollary form), computed on the *selected* signature
    size using q_lambda_estimate as q_Lambda (avg #selected on a random half)."""
    p = len(pi_hat)
    denom = (2 * pi_thr - 1)
    if denom <= 0:
        return np.inf
    return (q_lambda_estimate ** 2) / (denom * p)


def build_signatures(Z, y, pi_thr=0.6, B=25, seed=0):
    """Z: (n, m) clause activations on validation split. y: (n,) family index.

    S_l (the reported/certified signature, used for the E[V] bound) is the
    full sign-agnostic stability-selected support, per the doc's spec. The
    sign-split refinement: S_l_pos (inculpatory, mean_coef > 0) is evidence
    FOR family l; S_l_neg (exculpatory, mean_coef < 0) is evidence AGAINST
    it -- e.g. a clause that stably predicts "not family l" because it
    belongs to another family's clause bank. Both halves inherit the E[V]
    certificate from the sign-agnostic S_l (the S-S/MB bound controls false
    inclusions in the *selected* support regardless of sign; since
    S_pos, S_neg subset S_l, the certificate applies to their union).
    signatures[fam] stores {"pos": [...], "neg": [...]} for
    attribution_scores() below.
    """
    signatures = {}
    diagnostics = {}
    m = Z.shape[1]
    for cls_idx, fam in enumerate(FAMILIES):
        t0 = time.time()
        y_bin = (y == cls_idx).astype(int)
        pi_hat, mean_coef = cpss_proportions(Z, y_bin, B=B, seed=seed + cls_idx)
        S = np.where(pi_hat >= pi_thr)[0]
        S_pos = np.where((pi_hat >= pi_thr) & (mean_coef > 0))[0]
        S_neg = np.where((pi_hat >= pi_thr) & (mean_coef < 0))[0]
        # q_Lambda estimate: average number of clauses selected at the union-grid
        # per half-subsample (proxy = mean of selected_count-derived rate * m)
        q_est = float(np.mean(pi_hat) * m)
        eb = error_bound(pi_hat, pi_thr, q_est)
        signatures[fam] = {"pos": S_pos.tolist(), "neg": S_neg.tolist()}
        diagnostics[fam] = {
            "pi_hat": pi_hat.tolist(),
            "mean_coef": mean_coef.tolist(),
            "signature_size": int(len(S)),
            "signature_size_signed": int(len(S_pos)),
            "signature_size_exculpatory": int(len(S_neg)),
            "q_lambda_estimate": q_est,
            "E_V_bound": eb,
            "time_s": time.time() - t0,
            "prevalence": float(y_bin.mean()),
        }
        print(f"[{fam}] |S|={len(S)} |S_pos|={len(S_pos)} |S_neg|={len(S_neg)} "
              f"q~{q_est:.1f} E[V]<={eb:.3f} ({time.time()-t0:.1f}s)")
    return signatures, diagnostics


def attribution_scores(Z, signatures, diagnostics):
    """s_l(x) = [sum_{j in S_l_pos} pi_hat_j*z_j(x) - sum_{j in S_l_neg} pi_hat_j*z_j(x)]
       / [sum_{j in S_l_pos} pi_hat_j]
    for all families l: inculpatory evidence increases the score, exculpatory
    evidence (stably associated with NOT being family l) decreases it.
    Normalized by the inculpatory weight only, so a sample matching its full
    positive signature with no exculpatory evidence scores 1.0, and can go
    negative if exculpatory evidence dominates."""
    n = Z.shape[0]
    scores = np.zeros((n, len(FAMILIES)))
    for i, fam in enumerate(FAMILIES):
        S_pos = signatures[fam]["pos"] if isinstance(signatures[fam], dict) else signatures[fam]
        S_neg = signatures[fam].get("neg", []) if isinstance(signatures[fam], dict) else []
        pi_hat = np.array(diagnostics[fam]["pi_hat"])
        pos_denom = pi_hat[S_pos].sum() if len(S_pos) else 0.0
        if pos_denom <= 0:
            continue
        pos_term = (Z[:, S_pos] @ pi_hat[S_pos]) if len(S_pos) else 0.0
        neg_term = (Z[:, S_neg] @ pi_hat[S_neg]) if len(S_neg) else 0.0
        scores[:, i] = (pos_term - neg_term) / pos_denom
    return scores


if __name__ == "__main__":
    import pickle
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    pi_thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    with open(f"/FPTM/CAS_IoMT_Empirical/results/tm_model_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    d = np.load("/FPTM/CAS_IoMT_Empirical/results/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]

    print("extracting activations Z = tm.transform(Xte) ...")
    t0 = time.time()
    Z = tm.transform(Xte).astype(np.float64)
    print(f"Z shape {Z.shape}  ({time.time()-t0:.1f}s)")

    # split validation (for CPSS fitting) vs held-out eval (for Task A metrics)
    rng = np.random.RandomState(0)
    n = Z.shape[0]
    perm = rng.permutation(n)
    val_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    signatures, diagnostics = build_signatures(Z[val_idx], yte[val_idx], pi_thr=pi_thr, B=B, seed=seed)

    with open(f"/FPTM/CAS_IoMT_Empirical/results/stabilis_signatures_seed{seed}_thr{pi_thr}.json", "w") as f:
        json.dump({"signatures": signatures, "diagnostics": diagnostics,
                    "pi_thr": pi_thr, "B": B, "seed": seed,
                    "val_idx": val_idx.tolist(), "eval_idx": eval_idx.tolist()}, f)
    print("saved signatures.")
