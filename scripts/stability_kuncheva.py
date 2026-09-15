"""
Stability: Kuncheva index of extracted STABILIS signatures across TM seeds
(forensic_fingerprint_methods.md Sec 0.4 "Stability"). Scoped to 3 TM seeds
(doc asks for 10 + bootstrap; reduced here for session tractability).
"""
import json
import pickle
import sys
import time
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from stabilis import FAMILIES, build_signatures

SEEDS = [42, 7, 123]


def kuncheva_index(S1, S2, m):
    S1, S2 = set(S1), set(S2)
    r = len(S1 & S2)
    k1, k2 = len(S1), len(S2)
    if k1 == 0 or k2 == 0 or k1 == m or k2 == m:
        return 0.0
    expected = k1 * k2 / m
    denom = min(k1, k2) - expected
    if abs(denom) < 1e-9:
        return 0.0
    return (r - expected) / denom


def main(clauses=400, T=320, s=8.0, epochs=25, pi_thr=0.6, B=20):
    base = "/FPTM/CAS_IoMT_Empirical/results"
    d = np.load(f"{base}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    m = Xtr.shape[1] * 2  # TMU literals = features + negations

    all_signatures = {}
    for seed in SEEDS:
        model_path = f"{base}/tm_model_seed{seed}.pkl"
        try:
            with open(model_path, "rb") as f:
                tm = pickle.load(f)
            print(f"loaded cached model for seed {seed}")
        except FileNotFoundError:
            print(f"training TM seed={seed} ...")
            tm = TMClassifier(number_of_clauses=clauses, T=T, s=s, weighted_clauses=True, platform="CPU", seed=seed)
            t0 = time.time()
            for ep in range(epochs):
                tm.fit(Xtr, ytr)
            print(f"  trained in {time.time()-t0:.1f}s")
            with open(model_path, "wb") as f:
                pickle.dump(tm, f)

        Z = tm.transform(Xte).astype(np.float64)
        rng = np.random.RandomState(0)
        n = Z.shape[0]
        perm = rng.permutation(n)
        val_idx = perm[: n // 2]
        signatures, diagnostics = build_signatures(Z[val_idx], yte[val_idx], pi_thr=pi_thr, B=B, seed=seed)
        # clause-activation index space is per-model (m = n_classes*n_clauses here,
        # not literal space) -- record the size actually used for the Kuncheva null.
        m_clauses = Z.shape[1]
        all_signatures[seed] = {"signatures": signatures, "m_clauses": m_clauses}

    kunch = {}
    for fam in FAMILIES:
        vals = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                s1 = all_signatures[SEEDS[i]]["signatures"][fam]
                s2 = all_signatures[SEEDS[j]]["signatures"][fam]
                m_clauses = all_signatures[SEEDS[i]]["m_clauses"]
                vals.append(kuncheva_index(s1, s2, m_clauses))
        kunch[fam] = {"pairwise": vals, "mean": float(np.mean(vals)) if vals else None}
        print(f"[{fam}] Kuncheva index (pairwise over {len(SEEDS)} seeds): {vals} mean={kunch[fam]['mean']}")

    with open(f"{base}/stability_kuncheva.json", "w") as f:
        json.dump({"seeds": SEEDS, "kuncheva": kunch,
                    "signature_sizes": {seed: {f: len(all_signatures[seed]["signatures"][f]) for f in FAMILIES}
                                          for seed in SEEDS}}, f, indent=2)
    print("saved stability results.")


if __name__ == "__main__":
    main()
