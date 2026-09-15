"""
Where should CPSS get its labelled data? Three candidate splits, compared.

CAS fits twice: the TM, then the CPSS readout over that TM's clauses. A plain
train/val/test split forces the second fit to reuse data, and there are three
ways to arrange it. The TM is held fixed (trained on `core`) in every arm, and
the CPSS sample size is matched, so the ONLY variable is the source of CPSS's
labels.

  A   CPSS on `cpss`  -- held out from TM training          (4-way split)
  B   CPSS on `selm`, operating point ALSO chosen on `selm` (3-way, val reused)
  C   CPSS on `core`  -- data the TM itself trained on      (3-way, train reused)

A_full additionally shows what the extra CPSS data in the 4-way split buys.
Every arm reports once on the untouched full test set, 3 seeds.

Read the result for AGREEMENT, not for the maximum: an arm scoring above A is
optimistic, not better.

    python3 scripts/split_ablation.py
"""
import json, os, time
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import f1_score

import heldout_protocol as hp
import argparse

BASE, OUT = hp.BASE, hp.OUT
SEEDS, NCLS, NC = hp.SEEDS, hp.NCLS, hp.NC
FAMILIES = hp.FAMILIES
MATCH_N = None          # set to the smallest arm's size so source is the only variable


def sub(idx, y, n, seed=0):
    """Stratified subsample of `idx` down to n rows total."""
    if n is None or len(idx) <= n:
        return idx
    rng = np.random.RandomState(seed)
    frac = n / len(idx)
    keep = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = max(2, int(round(len(ci) * frac)))
        keep.append(ci if len(ci) <= k else rng.choice(ci, k, replace=False))
    return np.sort(np.concatenate(keep))


def _job(a):
    arm, seed, fi, Zpath, ybin, cols = a
    t0 = time.time()
    Z = np.asarray(np.load(Zpath), dtype=np.float32)[:, cols]
    out = hp.cpss_snap(Z, ybin, seed + fi)
    print(f"[cpss] {arm} seed{seed} {FAMILIES[fi]} {time.time()-t0:.0f}s", flush=True)
    return (arm, seed, fi, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="full", choices=["full", "pruned"])
    ap.add_argument("--keep", type=int, default=60, help="clauses kept per class when pruned")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    Xtr, ytr, Xte, yte = hp.load()
    core, cpss, selm = hp.train_split(ytr)
    MATCH = len(selm)          # smallest arm -> match everything to it
    arms = {
        "A_matched": sub(cpss, ytr, MATCH, 1),
        "A_full":    cpss,
        "B":         selm,
        "C":         sub(core, ytr, MATCH, 2),
    }
    for k, v in arms.items():
        print(f"[arm] {k:10s} CPSS rows={len(v)}", flush=True)

    # cache clause activations per arm per seed
    paths = {}
    for seed in SEEDS:
        import pickle
        with open(f"{OUT}/tm_core_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        for arm, idx in arms.items():
            p = f"{OUT}/Zsplit_{arm}_seed{seed}.npy"
            if not os.path.exists(p):
                np.save(p, tm.transform(Xtr[idx]).astype(np.uint8))
            paths[(arm, seed)] = p
        for nm, X in (("selm", Xtr[selm]), ("test", Xte)):
            p = f"{OUT}/Z_{nm}_seed{seed}.npy"
            if not os.path.exists(p):
                np.save(p, tm.transform(X).astype(np.uint8))
        print(f"[cache] seed{seed} done", flush=True)

    # pruned pool: keep only the highest-|weight| clauses per class
    cols = {}
    for seed in SEEDS:
        if args.pool == "pruned":
            W = np.load(f"{OUT}/W_seed{seed}.npy")
            cols[seed] = hp.keep_cols(W, args.keep)
        else:
            cols[seed] = np.arange(NC * NCLS)
    print(f"[pool] {args.pool}: {len(cols[SEEDS[0]])} clause columns", flush=True)

    jobs = [(arm, seed, fi, paths[(arm, seed)],
             (ytr[arms[arm]] == fi).astype(int), cols[seed])
            for arm in arms for seed in SEEDS for fi in range(NCLS)]
    print(f"[run] {len(jobs)} CPSS jobs", flush=True)
    with Pool(processes=min(36, len(jobs))) as p:
        res = p.map(_job, jobs)
    snaps = {(a, s, f): o for a, s, f, o in res}

    ysl = ytr[selm]
    out = {}
    for arm in arms:
        grid = {}
        for seed in SEEDS:
            Zs = np.asarray(np.load(f"{OUT}/Z_selm_seed{seed}.npy"), np.float32)[:, cols[seed]]
            Zt = np.asarray(np.load(f"{OUT}/Z_test_seed{seed}.npy"), np.float32)[:, cols[seed]]
            for B in hp.B_SNAPS:
                pis = [np.array(snaps[(arm, seed, i)][B][0]) for i in range(NCLS)]
                cfs = [np.array(snaps[(arm, seed, i)][B][1]) for i in range(NCLS)]
                for thr in hp.PI_THRS:
                    fs, _, _ = hp.score(Zs, ysl, pis, cfs, thr)
                    ft, at, sz = hp.score(Zt, yte, pis, cfs, thr)
                    grid.setdefault((B, thr), []).append(
                        {"seed": seed, "sel_f1": fs, "test_f1": ft,
                         "test_acc": at, "S_pos": float(np.mean(sz))})
        gm = {f"{B}|{t}": {"sel_f1": float(np.mean([r["sel_f1"] for r in v])),
                           "test_f1": float(np.mean([r["test_f1"] for r in v])),
                           "test_f1_sd": float(np.std([r["test_f1"] for r in v])),
                           "test_acc": float(np.mean([r["test_acc"] for r in v])),
                           "support": float(np.mean([r["S_pos"] for r in v]))}
              for (B, t), v in grid.items()}
        best = max(gm, key=lambda k: gm[k]["sel_f1"])
        out[arm] = {"cpss_rows": int(len(arms[arm])), "selected": best,
                    "result": gm[best], "grid": gm}
        r = gm[best]
        print(f"[{arm:10s}] n={len(arms[arm]):>6} selected={best:9s} "
              f"TEST f1={r['test_f1']:.4f}+-{r['test_f1_sd']:.4f} "
              f"acc={r['test_acc']:.4f} support={r['support']:.1f}", flush=True)
        json.dump(out, open(f"{OUT}/{args.out or 'split_ablation.json'}", "w"), indent=2)
    for arm in arms:
        for seed in SEEDS:
            f = f"{OUT}/Zsplit_{arm}_seed{seed}.npy"
            if os.path.exists(f):
                os.remove(f)
    print(f"[out] {OUT}/split_ablation.json", flush=True)


if __name__ == "__main__":
    main()
