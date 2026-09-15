"""kappa x pi_thr grid (Table II) on the leak-free three-way split.

Reuses full_cas.cpss_full, which records per-kappa AND union-over-grid
selection for every B snapshot, so single-kappa and the union rule come from
the identical set of half-fits. Selection quarter and report quarter are the
same ones threeway_cas.py uses. Full 2,400-clause pool.

Output: results/threeway_kappa.json
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import f1_score
sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from full_cas import KAPPA_GRID

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NCLS, NC = 6, 400
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B = 15


def splits(n):
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def cpss_perkappa(Z, ybin, seed):
    """Same half-fits as threeway_cas, but keeping per-kappa selection too."""
    from sklearn.linear_model import LogisticRegression
    n, m = Z.shape
    ng = len(KAPPA_GRID)
    rng = np.random.RandomState(seed)
    half = n // 2
    sel = np.zeros((ng, m), np.int64); coef = np.zeros((ng, m), np.float64)
    usel = np.zeros(m, np.int64);      ucoef = np.zeros(m, np.float64)
    for b in range(B):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            u = np.zeros(m, bool)
            for gi, C in enumerate(KAPPA_GRID):
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                         max_iter=200, class_weight="balanced",
                                         random_state=0)
                clf.fit(Zh, yh)
                cf = clf.coef_.ravel(); nz = np.abs(cf) > 1e-8
                sel[gi] += nz; coef[gi] += cf; u |= nz
                ucoef += cf
            usel += u
    d = 2 * B
    return ({gi: (sel[gi] / d, coef[gi] / d) for gi in range(ng)},
            (usel / d, ucoef / (d * ng)))


def job(a):
    seed, fi = a
    t0 = time.time()
    yte = np.load(f"{BASE}/booleanized_data.npz")["yte"]
    Z = np.load(f"{BASE}/Zte_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    Zf = np.asarray(Z[fit], dtype=np.float32)
    pk, un = cpss_perkappa(Zf, (yte[fit] == fi).astype(int), seed=seed + fi)
    print(f"[k] seed{seed} fam{fi} {time.time()-t0:.0f}s", flush=True)
    return (seed, fi,
            {str(gi): (v[0].tolist(), v[1].tolist()) for gi, v in pk.items()},
            (un[0].tolist(), un[1].tolist()))


def score(Zev, yev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], NCLS))
    for i in range(NCLS):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    return float(f1_score(yev, np.argmax(S, 1), average="macro"))


def main():
    yte = np.load(f"{BASE}/booleanized_data.npz")["yte"]
    fit, sel, rep = splits(len(yte))
    jobs = [(s, f) for s in SEEDS for f in range(NCLS)]
    with Pool(processes=18) as p:
        res = p.map(job, jobs)
    PK = {(s, f): pk for s, f, pk, _ in res}
    UN = {(s, f): un for s, f, _, un in res}

    out = {"kappa_grid": KAPPA_GRID.tolist(), "pi_thrs": PI_THRS, "B": B,
           "per_kappa_rep": {}, "per_kappa_sel": {}, "union_rep": {}, "union_sel": {}}
    Zc = {s: np.load(f"{BASE}/Zte_seed{s}.npy", mmap_mode="r") for s in SEEDS}
    for gi in list(range(len(KAPPA_GRID))) + ["union"]:
        for thr in PI_THRS:
            rs, ss = [], []
            for s in SEEDS:
                Zs = np.asarray(Zc[s][sel], dtype=np.float32)
                Zr = np.asarray(Zc[s][rep], dtype=np.float32)
                if gi == "union":
                    pis = [np.array(UN[(s, f)][0]) for f in range(NCLS)]
                    cfs = [np.array(UN[(s, f)][1]) for f in range(NCLS)]
                else:
                    pis = [np.array(PK[(s, f)][str(gi)][0]) for f in range(NCLS)]
                    cfs = [np.array(PK[(s, f)][str(gi)][1]) for f in range(NCLS)]
                ss.append(score(Zs, yte[sel], pis, cfs, thr))
                rs.append(score(Zr, yte[rep], pis, cfs, thr))
            key = f"{gi}|{thr}"
            (out["union_rep"] if gi == "union" else out["per_kappa_rep"])[key] = float(np.mean(rs))
            (out["union_sel"] if gi == "union" else out["per_kappa_sel"])[key] = float(np.mean(ss))
            print(f"  kappa={gi} thr={thr}: sel {np.mean(ss):.4f} rep {np.mean(rs):.4f}", flush=True)
    with open(f"{BASE}/threeway_kappa.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[done] threeway_kappa.json", flush=True)


if __name__ == "__main__":
    main()
