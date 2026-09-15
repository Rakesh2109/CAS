"""
Control: give the TM the SAME data CAS's CPSS is fitted on.

The paper's table compares a TM trained on Xtr (0.692) against CAS fitted on the
test fit-half (0.787). Those are different information conditions, so the gap is
not attributable to the method. This trains the TM directly on the fit half and
scores it on the report quarter -- the exact surface CAS reports on -- so TM and
CAS are finally measured under identical access.

If the TM lands near the fit-matched trees (~0.847), CAS is not improving on the
TM at all: it is losing to a TM trained on CAS's own fitting data.
"""
import json, time
import numpy as np
from sklearn.metrics import f1_score
from tmu.models.classification.vanilla_classifier import TMClassifier

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]


def splits(n):
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte, yte = d["Xte"], d["yte"]
    fit, sel, rep = splits(len(yte))
    print(f"[data] fit={len(fit)} sel={len(sel)} rep={len(rep)}", flush=True)
    out = []
    for seed in SEEDS:
        tm = TMClassifier(number_of_clauses=400, T=320, s=8.0,
                          weighted_clauses=True, platform="CPU", seed=seed)
        t0 = time.time()
        for _ in range(25):
            tm.fit(Xte[fit], yte[fit].astype(np.uint32))
        pr = tm.predict(Xte[rep])
        f1 = float(f1_score(yte[rep], pr, average="macro"))
        acc = float((pr == yte[rep]).mean())
        out.append({"seed": seed, "rep_f1": f1, "rep_acc": acc})
        print(f"[tm-fit] seed{seed} rep macro-F1={f1:.4f} acc={acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    f = [r["rep_f1"] for r in out]; a = [r["rep_acc"] for r in out]
    res = {"tm_trained_on_fit_half": {
        "rep_f1_mean": float(np.mean(f)), "rep_f1_sd": float(np.std(f)),
        "rep_acc_mean": float(np.mean(a)), "rep_acc_sd": float(np.std(a)),
        "per_seed": out}}
    json.dump(res, open(f"{BASE}/tm_fitmatched.json", "w"), indent=2)
    print(f"[done] TM fit-matched rep macro-F1={np.mean(f):.4f}+-{np.std(f):.4f} "
          f"acc={np.mean(a):.4f}", flush=True)


if __name__ == "__main__":
    main()
