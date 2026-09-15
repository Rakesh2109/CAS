"""
Table-I-style binary results (TM/CAS x full/pruned pool) on the RETUNED
binary TM from binary_tm_gridsearch.py.

Identical protocol to threeway_binary_pruned.py -- magnitude pruning with the
epsilon=0.01 non-degradation rule chosen on the selection quarter, CPSS on the
fit half for both pools, (B, pi_thr) chosen on the selection quarter, report
quarter read once -- but pointed at the retuned weights/activations so the
four rows describe a detector that is not prior-collapsed.

Usage: python3 scripts/binary_retuned_pruned.py [--tag _robust]
Output: results/binary_retuned_pruned{tag}.json
"""
import argparse, json, os, time
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "Attack"]
SEEDS = [42, 7, 123]
NCLS = 2
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25
EPS = 0.01
STEP = 5
TAG = "_robust"
NC = None


def splits(n):
    p = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return p[:h], p[h:h + q], p[h + q:]


def prep():
    return (np.load(f"{BASE}/booleanized_data.npz")["yte"] != 0).astype(np.uint8)


def class_sums(Z, W, cpc):
    return np.column_stack([Z[:, c * cpc:(c + 1) * cpc] @ W[c * cpc:(c + 1) * cpc]
                            for c in range(NCLS)])


def keep_cols(W, keep):
    idx = []
    for c in range(NCLS):
        o = np.argsort(np.abs(W[c * NC:(c + 1) * NC]), kind="stable")
        idx.append(c * NC + np.sort(o[NC - keep:]))
    return np.concatenate(idx)


def cpss_snap(Z, ybin, seed):
    n, m = Z.shape
    rng = np.random.RandomState(seed)
    half = n // 2
    sel = np.zeros(m, np.int64); csum = np.zeros(m, np.float64)
    ng = len(KAPPA_GRID)
    out = {}
    for b in range(BMAX):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Zh, yh = Z[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            u = np.zeros(m, bool)
            for C in KAPPA_GRID:
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                         max_iter=200, class_weight="balanced",
                                         random_state=0)
                clf.fit(Zh, yh)
                cf = clf.coef_.ravel()
                u |= np.abs(cf) > 1e-8
                csum += cf
            sel += u
        if (b + 1) in B_SNAPS:
            out[b + 1] = ((sel / (2 * (b + 1))).tolist(),
                          (csum / (2 * (b + 1) * ng)).tolist())
    return out


def job(args):
    seed, pool, fam_i, cols = args
    t0 = time.time()
    yb = prep()
    Z = np.load(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    Zf = np.asarray(Z[fit][:, cols], dtype=np.float32)
    cache = f"{BASE}/binary_retuned_snaps{TAG}_{pool}_seed{seed}_class{fam_i}.json"
    if os.path.exists(cache):
        out = {int(k): v for k, v in json.load(open(cache)).items()}
    else:
        out = cpss_snap(Zf, (yb[fit] == fam_i).astype(int), seed=seed + fam_i)
        with open(cache, "w") as f:
            json.dump(out, f)
    print(f"[cpss] seed{seed} {pool} {FAMILIES[fam_i]} {time.time()-t0:.0f}s", flush=True)
    return (seed, pool, fam_i, out)


def score(Zev, yev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], NCLS))
    sp, sa = [], []
    for i in range(NCLS):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        sp.append(len(Sp)); sa.append(int((pi >= thr).sum()))
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    pred = np.argmax(S, 1)
    return (float(f1_score(yev, pred, average="macro")), float((pred == yev).mean()),
            sp, sa, float(roc_auc_score(yev, S[:, 1] - S[:, 0])))


def main():
    global TAG, NC
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_robust")
    TAG = ap.parse_args().tag
    cfg = json.load(open(f"{BASE}/binary_retuned{TAG}.json"))["config"]
    NC = cfg["clauses"]
    yb = prep()
    fit, sel, rep = splits(len(yb))
    print(f"[cfg] {cfg}  NC={NC}  pool={NC*NCLS}", flush=True)

    # stage 1: pruning budget on the selection quarter
    ks = list(range(0, NC, STEP))
    csel, crep = [], []
    for seed in SEEDS:
        Z = np.load(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy", mmap_mode="r")
        W = np.load(f"{BASE}/W_binary_retuned{TAG}_seed{seed}.npy")
        Zs = np.asarray(Z[sel], np.float32); Zr = np.asarray(Z[rep], np.float32)
        aw = np.abs(W)
        order = {c: c * NC + np.argsort(aw[c * NC:(c + 1) * NC], kind="stable")
                 for c in range(NCLS)}
        a, b = [], []
        for k in ks:
            Wm = W.copy()
            if k:
                Wm[np.concatenate([order[c][:k] for c in range(NCLS)])] = 0.0
            a.append(float(f1_score(yb[sel], np.argmax(class_sums(Zs, Wm, NC), 1), average="macro")))
            b.append(float(f1_score(yb[rep], np.argmax(class_sums(Zr, Wm, NC), 1), average="macro")))
        csel.append(a); crep.append(b)
    msel = np.array(csel).mean(0)
    ok = [i for i, v in enumerate(msel) if v >= msel[0] - EPS]
    kstar = ks[max(ok)]; keep = NC - kstar
    print(f"[prune] k*={kstar}/class -> keep {keep}/class, pool {keep*NCLS} "
          f"(sel F1 {msel[0]:.4f} -> {msel[max(ok)]:.4f})", flush=True)

    pools = {}
    for seed in SEEDS:
        W = np.load(f"{BASE}/W_binary_retuned{TAG}_seed{seed}.npy")
        pools[(seed, "full")] = np.arange(NC * NCLS)
        pools[(seed, "pruned")] = keep_cols(W, keep)
    jobs = [(s, po, fi, cols) for (s, po), cols in pools.items() for fi in range(NCLS)]
    with Pool(processes=min(12, len(jobs))) as p:
        res = p.map(job, jobs)
    snaps = {(s, po, fi): o for s, po, fi, o in res}

    out = {"config": cfg, "kstar_removed_per_class": kstar, "keep_per_class": keep,
           "pool_pruned": keep * NCLS, "ks": ks, "prune_curve_sel": csel,
           "prune_curve_rep": crep, "n": {"fit": len(fit), "sel": len(sel), "rep": len(rep)},
           "pools": {}}
    for pool in ("full", "pruned"):
        cpc = NC if pool == "full" else keep
        bs, br, ba_s, ba_r, bau, grid = [], [], [], [], [], {}
        for seed in SEEDS:
            Z = np.load(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy", mmap_mode="r")
            W = np.load(f"{BASE}/W_binary_retuned{TAG}_seed{seed}.npy")
            cols = pools[(seed, pool)]
            Zs = np.asarray(Z[sel][:, cols], np.float32)
            Zr = np.asarray(Z[rep][:, cols], np.float32)
            Wk = W[cols]
            ps = np.argmax(class_sums(Zs, Wk, cpc), 1); pr = np.argmax(class_sums(Zr, Wk, cpc), 1)
            bs.append(float(f1_score(yb[sel], ps, average="macro")))
            br.append(float(f1_score(yb[rep], pr, average="macro")))
            ba_s.append(float((ps == yb[sel]).mean())); ba_r.append(float((pr == yb[rep]).mean()))
            csr = class_sums(Zr, Wk, cpc)
            bau.append(float(roc_auc_score(yb[rep], csr[:, 1] - csr[:, 0])))
            for Bv in B_SNAPS:
                pis = [np.array(snaps[(seed, pool, i)][Bv][0]) for i in range(NCLS)]
                cfs = [np.array(snaps[(seed, pool, i)][Bv][1]) for i in range(NCLS)]
                for thr in PI_THRS:
                    fs, as_, sp, sa, _ = score(Zs, yb[sel], pis, cfs, thr)
                    fr, ar, spr, sar, aur = score(Zr, yb[rep], pis, cfs, thr)
                    grid.setdefault((Bv, thr), []).append(
                        {"seed": seed, "sel_f1": fs, "sel_acc": as_, "rep_f1": fr,
                         "rep_acc": ar, "rep_auroc": aur, "S_pos": sp, "S": sa})
        gm = {f"{B}|{t}": {"sel_f1": float(np.mean([r["sel_f1"] for r in v])),
                           "rep_f1": float(np.mean([r["rep_f1"] for r in v])),
                           "rep_f1_sd": float(np.std([r["rep_f1"] for r in v])),
                           "rep_acc": float(np.mean([r["rep_acc"] for r in v])),
                           "rep_acc_sd": float(np.std([r["rep_acc"] for r in v])),
                           "rep_auroc": float(np.mean([r["rep_auroc"] for r in v])),
                           "rep_auroc_sd": float(np.std([r["rep_auroc"] for r in v])),
                           "mean_S_pos": float(np.mean([np.mean(r["S_pos"]) for r in v])),
                           "per_seed": v} for (B, t), v in grid.items()}
        best = max(gm, key=lambda k: gm[k]["sel_f1"])
        Bb, tb = best.split("|")
        out["pools"][pool] = {
            "clauses_per_class": cpc, "pool": cpc * NCLS,
            "base_sel_f1": float(np.mean(bs)), "base_rep_f1": float(np.mean(br)),
            "base_rep_f1_sd": float(np.std(br)),
            "base_rep_acc": float(np.mean(ba_r)), "base_rep_acc_sd": float(np.std(ba_r)),
            "base_rep_auroc": float(np.mean(bau)), "base_rep_auroc_sd": float(np.std(bau)),
            "selected_B": int(Bb), "selected_pi_thr": float(tb),
            "reported": gm[best], "grid": gm}
        p_ = out["pools"][pool]
        print(f"[{pool}] TM rep F1 {p_['base_rep_f1']:.4f}+/-{p_['base_rep_f1_sd']:.4f} "
              f"acc {p_['base_rep_acc']:.4f} | CAS (B={Bb},thr={tb}) rep F1 "
              f"{gm[best]['rep_f1']:.4f}+/-{gm[best]['rep_f1_sd']:.4f} "
              f"acc {gm[best]['rep_acc']:.4f}+/-{gm[best]['rep_acc_sd']:.4f}", flush=True)

    with open(f"{BASE}/binary_retuned_pruned{TAG}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] results/binary_retuned_pruned{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
