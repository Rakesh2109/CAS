"""
CAS applied to tree-derived rules instead of TM clauses.

The CAS pipeline has two separable parts: (1) a pool of Boolean conjunctions
over the input literals, and (2) a CPSS readout that stability-selects a signed
subset of that pool and scores classes by a normalised signed sum. Part (2) is
substrate-agnostic. This script keeps part (2) bit-identical and swaps part (1)
from TM clauses to random-forest leaves.

Each leaf of a decision tree IS a conjunction of literal tests, i.e. structurally
the same object as a TM clause. A forest of 50 trees capped at 48 leaves yields
~2,400 rules, matching the TM's 400 clauses x 6 classes = 2,400 clause pool, so
the two evidence pools are the same size and the comparison isolates the
question the reviewer is really asking: are TM clauses better evidence
primitives than tree rules?

Protocol is the leak-free one from heldout_protocol.py -- rules are grown on
`core`, CPSS is fitted on `cpss`, the operating point is chosen on `selm`, and
the test set is touched once.

    python3 scripts/rule_cas.py --forest 50x48
"""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import heldout_protocol as hp

BASE = hp.BASE
OUT = hp.OUT
SEEDS = hp.SEEDS
NCLS = hp.NCLS
FAMILIES = hp.FAMILIES


def build_rules(Xc, yc, seed, n_trees, max_leaves):
    """Grow a forest and return a transformer mapping X -> binary leaf indicators."""
    rf = RandomForestClassifier(n_estimators=n_trees, max_leaf_nodes=max_leaves,
                                class_weight="balanced_subsample",
                                n_jobs=hp_njobs, random_state=seed)
    rf.fit(Xc, yc)
    # leaf id -> column offset per tree
    offsets, total = [], 0
    for est in rf.estimators_:
        n_nodes = est.tree_.node_count
        offsets.append((total, n_nodes))
        total += n_nodes
    def transform(X, batch=20000):
        out = np.zeros((X.shape[0], total), dtype=np.uint8)
        for i in range(0, X.shape[0], batch):
            sl = slice(i, min(i + batch, X.shape[0]))
            leaves = rf.apply(X[sl])           # (n, n_trees) node ids
            for t, (off, _) in enumerate(offsets):
                out[sl, off + leaves[:, t]] = 1
        return out
    return rf, transform, total


def _job(a):
    seed, fi, Zpath, ycp = a
    t0 = time.time()
    Z = np.load(Zpath, mmap_mode="r")
    Zf = np.asarray(Z, dtype=np.float32)
    out = hp.cpss_snap(Zf, (ycp == fi).astype(int), seed + fi)
    print(f"[cpss] seed{seed} {FAMILIES[fi]} {time.time()-t0:.0f}s", flush=True)
    return (seed, fi, out)


def main():
    global hp_njobs
    ap = argparse.ArgumentParser()
    ap.add_argument("--forest", default="50x48", help="<n_trees>x<max_leaves>")
    ap.add_argument("--njobs", type=int, default=32)
    ap.add_argument("--out", default="rule_cas.json")
    a = ap.parse_args()
    hp_njobs = a.njobs
    n_trees, max_leaves = (int(v) for v in a.forest.split("x"))

    Xtr, ytr, Xte, yte = hp.load()
    core, cpss, selm = hp.train_split(ytr)
    Xc, yc = Xtr[core].astype(np.float32), ytr[core]
    ycp, ysl = ytr[cpss], ytr[selm]

    res = {"forest": a.forest, "seeds": SEEDS, "per_seed": []}
    grid_all = {}
    for seed in SEEDS:
        t0 = time.time()
        rf, transform, npool = build_rules(Xc, yc, seed, n_trees, max_leaves)
        print(f"[rules] seed{seed} pool={npool} rules ({time.time()-t0:.0f}s)", flush=True)
        # rf.apply only ever lights up leaf columns; drop the always-zero
        # internal-node columns so the CPSS matrix is the true 2,400-rule pool
        Zc = transform(Xtr[cpss].astype(np.float32))
        active = np.where(Zc.any(axis=0))[0]
        print(f"[rules] seed{seed} active leaf columns: {len(active)} of {npool}", flush=True)
        paths = {}
        for k, Z in (("cpss", Zc),
                     ("selm", transform(Xtr[selm].astype(np.float32))),
                     ("test", transform(Xte.astype(np.float32)))):
            p = f"{OUT}/Zrule_{k}_seed{seed}.npy"
            np.save(p, Z[:, active])
            paths[k] = p
        del Zc
        # forest's own argmax baseline, for reference
        pt = rf.predict(Xte.astype(np.float32))
        rf_f1 = float(f1_score(yte, pt, average="macro"))
        print(f"[rules] seed{seed} forest TEST macro-F1={rf_f1:.4f}", flush=True)

        jobs = [(seed, fi, paths["cpss"], ycp) for fi in range(NCLS)]
        try:
            with Pool(processes=min(NCLS, a.njobs)) as p:
                out = p.map(_job, jobs)
        except Exception:
            import traceback; traceback.print_exc()
            print("[warn] pool failed, running families sequentially", flush=True)
            out = [_job(j) for j in jobs]
        snaps = {fi: o for _, fi, o in out}

        Zs = np.asarray(np.load(paths["selm"]), np.float32)
        Zt = np.asarray(np.load(paths["test"]), np.float32)
        for Bv in hp.B_SNAPS:
            pis = [np.array(snaps[i][Bv][0]) for i in range(NCLS)]
            cfs = [np.array(snaps[i][Bv][1]) for i in range(NCLS)]
            for thr in hp.PI_THRS:
                fs, _, _ = hp.score(Zs, ysl, pis, cfs, thr)
                ft, at, sz = hp.score(Zt, yte, pis, cfs, thr)
                grid_all.setdefault((Bv, thr), []).append(
                    {"seed": seed, "selm_f1": fs, "test_f1": ft, "test_acc": at,
                     "S_pos": sz})
        res["per_seed"].append({"seed": seed, "n_rules": npool, "forest_test_f1": rf_f1})
        for k in paths.values():
            os.remove(k)

    gm = {f"{B}|{t}": {"selm_f1": float(np.mean([r["selm_f1"] for r in v])),
                       "test_f1": float(np.mean([r["test_f1"] for r in v])),
                       "test_f1_sd": float(np.std([r["test_f1"] for r in v])),
                       "test_acc": float(np.mean([r["test_acc"] for r in v])),
                       "test_acc_sd": float(np.std([r["test_acc"] for r in v])),
                       "mean_S_pos": float(np.mean([np.mean(r["S_pos"]) for r in v]))}
          for (B, t), v in grid_all.items()}
    best = max(gm, key=lambda k: gm[k]["selm_f1"])
    res["selected"] = best
    res["rule_cas"] = gm[best]
    res["grid"] = gm
    json.dump(res, open(f"{OUT}/{a.out}", "w"), indent=2)
    print(f"[rule-CAS] selected {best} -> TEST f1={gm[best]['test_f1']:.4f}"
          f"+-{gm[best]['test_f1_sd']:.4f} acc={gm[best]['test_acc']:.4f} "
          f"mean_support={gm[best]['mean_S_pos']:.1f}", flush=True)
    print(f"[out] {OUT}/{a.out}", flush=True)


if __name__ == "__main__":
    main()
