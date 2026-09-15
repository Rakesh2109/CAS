"""AUROC (and macro-F1) vs pi_thr for the binary CAS arm, B=15, 3 seeds.
Reuses the exact CPSS pass full_cas.py uses, just also scores AUROC per
threshold (which full_cas.py's B_sweep dict does not store)."""
import json
import pickle
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from full_cas import (BASE, FAM_BIN, stratified_subsample, cpss_full,
                      CAP_PER_CLASS, SUBSEED)

SEEDS = [42, 7, 123]
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]

d = np.load(f"{BASE}/dedup_bool_nbins5.npz")
Xte, yte = d["Xte"], d["yte"]
yte = (yte != 0).astype(np.uint8)

out = {"seeds": SEEDS, "pi_thrs": PI_THRS, "runs": []}
for seed in SEEDS:
    with open(f"{BASE}/tm_dedup_binary_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    sub = stratified_subsample(yte, CAP_PER_CLASS, SUBSEED)
    Zsub = tm.transform(np.ascontiguousarray(Xte[sub], np.uint32)).astype(np.float32)
    ysub = yte[sub]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(sub))
    val, ev = perm[:len(sub) // 2], perm[len(sub) // 2:]
    Zval, Zev, yval, yev = Zsub[val], Zsub[ev], ysub[val], ysub[ev]

    snaps = {}
    for ci, fname in enumerate(FAM_BIN):
        snaps[fname] = cpss_full(Zval, (yval == ci).astype(int), seed=seed + ci)

    row = {}
    for thr in PI_THRS:
        scores = np.full((len(ev), 2), -1e9)
        for ci, fname in enumerate(FAM_BIN):
            pi, cf = snaps[fname][15]["union"]
            Spos = np.where((pi >= thr) & (cf > 0))[0]
            Sneg = np.where((pi >= thr) & (cf < 0))[0]
            den = pi[Spos].sum()
            if den > 0:
                s = (Zev[:, Spos] @ pi[Spos] -
                    (Zev[:, Sneg] @ pi[Sneg] if len(Sneg) else 0)) / den
                scores[:, ci] = s
        pred = np.argmax(scores, 1)
        cas_score = scores[:, 1] - scores[:, 0]
        row[str(thr)] = {
            "macro_f1": float(f1_score(yev, pred, average="macro")),
            "accuracy": float((pred == yev).mean()),
            "auroc": float(roc_auc_score(yev, cas_score)),
        }
        print(f"[seed {seed}] thr={thr}: macroF1={row[str(thr)]['macro_f1']:.3f} "
              f"AUROC={row[str(thr)]['auroc']:.3f}", flush=True)
    out["runs"].append({"seed": seed, "pi_thr_sweep": row})

with open(f"{BASE}/dedup_binary_pithr_auroc.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved dedup_binary_pithr_auroc.json")
