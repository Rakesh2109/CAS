import argparse
import json
import pickle
import sys
import time
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
import stabilis as stab_mod
from stabilis import build_signatures

SEEDS = [42, 7, 123]

DATASET_CONFIG = {
    "medsec": {"dir": "results_medsec",
               "families": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
               "clauses": 400, "T": 320, "s": 8.0, "epochs": 25},
    "wustl": {"dir": "results_wustl",
              "families": ["normal", "Spoofing", "Data Alteration"],
              "clauses": 200, "T": 160, "s": 8.0, "epochs": 40},
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--pi_thr", type=float, default=0.6)
    ap.add_argument("--B", type=int, default=15)
    args = ap.parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{cfg['dir']}"
    families = cfg["families"]
    stab_mod.FAMILIES = families

    d = np.load(f"{base}/booleanized_data.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    all_signatures = {}
    for seed in SEEDS:
        model_path = f"{base}/tm_model_seed{seed}.pkl"
        try:
            with open(model_path, "rb") as f:
                tm = pickle.load(f)
            print(f"loaded cached model for seed {seed}")
        except FileNotFoundError:
            print(f"training TM seed={seed} ...")
            tm = TMClassifier(number_of_clauses=cfg["clauses"], T=cfg["T"], s=cfg["s"],
                               weighted_clauses=True, platform="CPU", seed=seed)
            t0 = time.time()
            for ep in range(cfg["epochs"]):
                tm.fit(Xtr, ytr)
            print(f"  trained in {time.time()-t0:.1f}s")
            with open(model_path, "wb") as f:
                pickle.dump(tm, f)

        Z = tm.transform(Xte).astype(np.float64)
        rng = np.random.RandomState(0)
        n = Z.shape[0]
        perm = rng.permutation(n)
        val_idx = perm[: n // 2]
        signatures, diagnostics = build_signatures(Z[val_idx], yte[val_idx], pi_thr=args.pi_thr, B=args.B, seed=seed)
        all_signatures[seed] = {"signatures": signatures, "m_clauses": Z.shape[1]}

    kunch = {}
    for fam in families:
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
                    "signature_sizes": {seed: {f: len(all_signatures[seed]["signatures"][f]) for f in families}
                                          for seed in SEEDS}}, f, indent=2)
    print("saved stability results for", args.dataset)


if __name__ == "__main__":
    main()
