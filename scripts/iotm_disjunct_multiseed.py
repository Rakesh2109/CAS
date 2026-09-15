"""
DISJUNCT (Method 3) on IoTM, multi-seed, on the nbins10 per-class signatures.

Reuses the cached nbins10 multiclass TMs from iotm_multiseed_perclass.py and
recomputes the CPSS signatures deterministically (seeded liblinear), then runs
the full DISJUNCT analysis per seed at pi_thr in {0.6, 0.8}:
  - t-disjunctness certificate (exact, brute force over the 5 other columns)
  - greedy overlap reduction
  - t=2 mixture-decoding simulation (noiseless strict COMP; noisy with
    thresholded COMP), per forensic_fingerprint_methods.md METHOD 3.

Output: results/iotm_disjunct_multiseed.json
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod
import disjunct as dis

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
PI_THRS = [0.6, 0.8]
stab_mod.FAMILIES = FAMILIES


def signatures_for_seed(seed, Z_val, y_val):
    """CPSS per family, identical settings to iotm_multiseed_perclass.py."""
    pi_hats, mean_coefs = {}, {}
    for cls_idx, fam in enumerate(FAMILIES):
        y_bin = (y_val == cls_idx).astype(int)
        pi_hat, mean_coef = stab_mod.cpss_proportions(Z_val, y_bin, B=15, seed=seed + cls_idx)
        pi_hats[fam], mean_coefs[fam] = pi_hat, mean_coef
        print(f"  [{fam}] CPSS done", flush=True)
    return pi_hats, mean_coefs


def disjunct_at_thr(pi_hats, mean_coefs, thr, m, seed):
    B = np.zeros((m, len(FAMILIES)), dtype=bool)
    for i, fam in enumerate(FAMILIES):
        B[:, i] = (pi_hats[fam] >= thr) & (mean_coefs[fam] > 0)
    sig_sizes = {fam: int(B[:, i].sum()) for i, fam in enumerate(FAMILIES)}

    t_raw, covers_raw = dis.t_disjunct_certificate(B, FAMILIES)
    B_red, n_removed = dis.greedy_overlap_reduction(B, FAMILIES)
    t_red, covers_red = dis.t_disjunct_certificate(B_red, FAMILIES)

    out = {"pi_thr": thr, "signature_sizes": sig_sizes,
           "t_hat_raw": t_raw, "covers_raw": covers_raw,
           "t_hat_reduced": t_red, "covers_reduced": covers_red,
           "n_clauses_removed": n_removed}
    for tag, Bmat in [("raw", B), ("reduced", B_red)]:
        for noise_p, comp_thr in [(0.0, 1.0), (0.05, 1.0), (0.05, 0.9), (0.05, 0.8), (0.15, 0.8)]:
            acc, _ = dis.simulate_mixture_decoding(Bmat, FAMILIES, t=2, n_trials=500,
                                                   seed=seed, noise_p=noise_p, comp_threshold=comp_thr)
            out[f"mixture_acc_{tag}_noise{noise_p}_thr{comp_thr}"] = acc
    return out


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    rng = np.random.RandomState(0)
    n = len(yte)
    perm = rng.permutation(n)
    val_idx = perm[: n // 2]

    results = {}
    for seed in SEEDS:
        print(f"\n===== seed {seed} =====", flush=True)
        with open(f"{BASE}/tm_model_nbins10_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        Z = tm.transform(Xte).astype(np.float64)
        m = Z.shape[1]
        pi_hats, mean_coefs = signatures_for_seed(seed, Z[val_idx], yte[val_idx])
        results[str(seed)] = {}
        for thr in PI_THRS:
            r = disjunct_at_thr(pi_hats, mean_coefs, thr, m, seed)
            results[str(seed)][f"thr{thr}"] = r
            print(f"  thr={thr}: t_hat raw={r['t_hat_raw']} reduced={r['t_hat_reduced']} "
                  f"sizes={r['signature_sizes']}", flush=True)
            print(f"    mixture t=2 noiseless strict-COMP: raw={r['mixture_acc_raw_noise0.0_thr1.0']:.3f} "
                  f"reduced={r['mixture_acc_reduced_noise0.0_thr1.0']:.3f}; "
                  f"noise5% thr0.9: {r['mixture_acc_raw_noise0.05_thr0.9']:.3f}", flush=True)

    with open(f"{BASE}/iotm_disjunct_multiseed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/iotm_disjunct_multiseed.json")


if __name__ == "__main__":
    main()
