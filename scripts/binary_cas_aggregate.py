"""
Aggregate binary-CAS results across seeds: mean/std metrics per dataset and
pairwise Kuncheva index of the ATTACK signature across seeds (same stability
criterion as the per-class pipeline's stability_kuncheva.py).
"""
import itertools
import json
import sys

import numpy as np

CONFIG = {
    "iotm": "results",
    "medsec": "results_medsec",
    "wustl": "results_wustl",
}
SEEDS = [42, 7, 123]


def kuncheva(S1, S2, m):
    """Kuncheva index for two subsets of {0..m-1}."""
    a = len(set(S1) & set(S2))
    n1, n2 = len(S1), len(S2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    expected = n1 * n2 / m
    denom = min(n1, n2) - expected
    if denom == 0:
        return float("nan")
    return (a - expected) / denom


def main():
    summary = {}
    for ds, base in CONFIG.items():
        runs = []
        for s in SEEDS:
            try:
                with open(f"/FPTM/CAS_IoMT_Empirical/{base}/binary_cas_results_seed{s}_thr0.6.json") as f:
                    runs.append(json.load(f))
            except FileNotFoundError:
                print(f"[{ds}] seed {s} missing, skipped")
        if not runs:
            continue
        m = None  # clause-pool width differs per seed's TM? No -- same config, same m.
        sigs = {}
        for r in runs:
            sigs[r["seed"]] = r["signatures"]["attack"]["pos"]
        # m = total clause outputs = 2 * clauses (normal+attack banks)
        m = 2 * runs[0]["tm_config"]["clauses"]
        ks = [kuncheva(sigs[a], sigs[b], m) for a, b in itertools.combinations(sigs, 2)]

        def agg(fn):
            vals = [fn(r) for r in runs]
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}

        summary[ds] = {
            "n_seeds": len(runs),
            "tm_attack_f1": agg(lambda r: r["tm_argmax"]["attack_f1"]),
            "tm_attack_recall": agg(lambda r: r["tm_argmax"]["attack_recall"]),
            "tm_auroc": agg(lambda r: r["tm_argmax"]["auroc"]),
            "cas_attack_f1": agg(lambda r: r["cas_attribution"]["attack_f1"]),
            "cas_attack_recall": agg(lambda r: r["cas_attribution"]["attack_recall"]),
            "cas_attack_precision": agg(lambda r: r["cas_attribution"]["attack_precision"]),
            "cas_auroc": agg(lambda r: r["cas_attribution"]["auroc"]),
            "cas_auprc": agg(lambda r: r["cas_attribution"]["auprc"]),
            "attack_sig_size": agg(lambda r: r["signature_summary"]["attack"]["signature_size_signed"]),
            "attack_E_V_bound": agg(lambda r: r["signature_summary"]["attack"]["E_V_bound"]),
            "attack_prevalence": runs[0]["attack_prevalence_test"],
            "attack_sig_kuncheva_pairwise": ks,
            "attack_sig_kuncheva_mean": float(np.nanmean(ks)),
        }
        print(f"[{ds}] TM attack-F1 {summary[ds]['tm_attack_f1']['mean']:.4f}±{summary[ds]['tm_attack_f1']['std']:.4f} | "
              f"CAS attack-F1 {summary[ds]['cas_attack_f1']['mean']:.4f}±{summary[ds]['cas_attack_f1']['std']:.4f} | "
              f"CAS AUROC {summary[ds]['cas_auroc']['mean']:.4f} | Kuncheva {summary[ds]['attack_sig_kuncheva_mean']:.3f}")

    with open("/FPTM/CAS_IoMT_Empirical/results/binary_cas_aggregate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/binary_cas_aggregate.json")


if __name__ == "__main__":
    main()
