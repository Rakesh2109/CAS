"""Cross-seed Kuncheva index on CAS inculpatory clause-index overlap,
full-dataset multiclass models (B=15 union grid, pi_thr=0.8)."""
import json, pickle, sys
import numpy as np
sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from full_cas import (BASE, FAM_MC, stratified_subsample, cpss_full, CAP_PER_CLASS, SUBSEED)

SEEDS = [42, 7, 123]
d = np.load(f"{BASE}/dedup_bool_nbins10.npz")
Xte, yte = d["Xte"], d["yte"]
sub = stratified_subsample(yte, CAP_PER_CLASS, SUBSEED)
ysub = yte[sub]
rng = np.random.RandomState(0); perm = rng.permutation(len(sub))
val = perm[:len(sub) // 2]

sigs = {s: {} for s in SEEDS}
for s in SEEDS:
    with open(f"{BASE}/tm_dedup_multiclass_seed{s}.pkl", "rb") as f:
        tm = pickle.load(f)
    Zval = tm.transform(np.ascontiguousarray(Xte[sub][val], np.uint32)).astype(np.float32)
    p = ysub[val]
    for ci, fam in enumerate(FAM_MC):
        snaps = cpss_full(Zval, (p == ci).astype(int), seed=s + ci)
        pi, cf = snaps[15]["union"]
        sigs[s][fam] = set(np.where((pi >= 0.8) & (cf > 0))[0].tolist())
    print("seed", s, "done", flush=True)

m = 2400
def kuncheva(A, B, m):
    n = len(A | B and A) if False else None
    r = len(A & B); k1, k2 = len(A), len(B)
    if k1 == 0 or k2 == 0 or k1 == m or k2 == m:
        return 0.0
    exp = k1 * k2 / m
    denom = (k1 + k2) / 2 - exp
    return 0.0 if denom == 0 else (r - exp) / denom

out = {}
for fam in FAM_MC:
    ps = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            ps.append(kuncheva(sigs[SEEDS[i]][fam], sigs[SEEDS[j]][fam], m))
    out[fam] = {"pairwise": ps, "mean": float(np.mean(ps))}
    print(f"{fam}: mean Kuncheva {np.mean(ps):+.3f}  sizes "
          f"{[len(sigs[s][fam]) for s in SEEDS]}", flush=True)
json.dump({"seeds": SEEDS, "kuncheva": out}, open(f"{BASE}/dedup_kuncheva.json", "w"), indent=2)
print("saved dedup_kuncheva.json")
