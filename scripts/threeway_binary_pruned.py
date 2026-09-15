"""Binary full/pruned reanalysis for CAS_4page tables; same selection protocol as threeway_cas.py. Saves CPSS snapshots for auditing."""
import json, os, pickle, time
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "Attack"]
SEEDS = [42, 7, 123]
NC, NCLS = 400, 2
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25
EPS = 0.01
STEP = 5


def splits(n):
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def prep():
    return (np.load(f"{BASE}/booleanized_data.npz")["yte"] != 0).astype(np.uint8)


def class_sums(Z, W, cols_per_class):
    return np.column_stack([Z[:, c * cols_per_class:(c + 1) * cols_per_class]
                            @ W[c * cols_per_class:(c + 1) * cols_per_class]
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
    sel = np.zeros(m, np.int64)
    csum = np.zeros(m, np.float64)
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
    yte = prep()
    Z = np.load(f"{BASE}/Zte_binary_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    Zf = np.asarray(Z[fit][:, cols], dtype=np.float32)
    yb = (yte[fit] == fam_i).astype(int)
    cache = f"{BASE}/binary_revision_snaps_{pool}_seed{seed}_class{fam_i}.json"
    if os.path.exists(cache):
        with open(cache) as f:
            out = {int(k): v for k, v in json.load(f).items()}
    else:
        out = cpss_snap(Zf, yb, seed=seed + fam_i)
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
    return (float(f1_score(yev, pred, average="macro")),
            float((pred == yev).mean()), sp, sa)


def main():
    yte = prep()
    n = len(yte)
    fit, sel, rep = splits(n)
    print(f"[split] fit={len(fit)} sel={len(sel)} rep={len(rep)}", flush=True)

    # ---------- stage 1: pruning budget chosen on the SELECTION quarter ----------
    ks = list(range(0, NC, STEP))
    curves_sel, curves_rep = [], []
    for seed in SEEDS:
        Z = np.load(f"{BASE}/Zte_binary_seed{seed}.npy", mmap_mode="r")
        W = np.load(f"{BASE}/W_binary_seed{seed}.npy")
        Zs = np.asarray(Z[sel], dtype=np.float32)
        Zr = np.asarray(Z[rep], dtype=np.float32)
        aw = np.abs(W)
        order = {c: c * NC + np.argsort(aw[c * NC:(c + 1) * NC], kind="stable")
                 for c in range(NCLS)}
        cs, cr = [], []
        for k in ks:
            Wm = W.copy()
            if k:
                Wm[np.concatenate([order[c][:k] for c in range(NCLS)])] = 0.0
            cs.append(float(f1_score(yte[sel], np.argmax(class_sums(Zs, Wm, NC), 1),
                                     average="macro")))
            cr.append(float(f1_score(yte[rep], np.argmax(class_sums(Zr, Wm, NC), 1),
                                     average="macro")))
        curves_sel.append(cs); curves_rep.append(cr)
        print(f"[prune] seed {seed} swept", flush=True)
    msel = np.array(curves_sel).mean(0)
    ok = [i for i, v in enumerate(msel) if v >= msel[0] - EPS]
    kstar = ks[max(ok)]
    keep = NC - kstar
    print(f"[prune] k*={kstar} removed/class -> keep {keep}/class, "
          f"pool {keep*NCLS}  (sel F1 {msel[0]:.4f} -> {msel[max(ok)]:.4f})", flush=True)

    # ---------- stage 2: CPSS on the fit half, both pools ----------
    jobs = []
    pools = {}
    for seed in SEEDS:
        W = np.load(f"{BASE}/W_binary_seed{seed}.npy")
        pools[(seed, "full")] = np.arange(NC * NCLS)
        pools[(seed, "pruned")] = keep_cols(W, keep)
    for (seed, pool), cols in pools.items():
        for fi in range(NCLS):
            jobs.append((seed, pool, fi, cols))
    with Pool(processes=min(6, len(jobs))) as p:
        res = p.map(job, jobs)
    snaps = {(s, po, fi): o for s, po, fi, o in res}

    # ---------- stage 3: select on sel, report once on rep ----------
    out = {"kstar_removed_per_class": kstar, "keep_per_class": keep,
           "pool_pruned": keep * NCLS, "ks": ks,
           "prune_curve_sel": curves_sel, "prune_curve_rep": curves_rep,
           "n": {"fit": len(fit), "sel": len(sel), "rep": len(rep)},
           "pools": {}}
    for pool in ("full", "pruned"):
        cpc = NC if pool == "full" else keep
        base_rep, base_sel = [], []
        grid = {}
        for seed in SEEDS:
            Z = np.load(f"{BASE}/Zte_binary_seed{seed}.npy", mmap_mode="r")
            W = np.load(f"{BASE}/W_binary_seed{seed}.npy")
            cols = pools[(seed, pool)]
            Zs = np.asarray(Z[sel][:, cols], dtype=np.float32)
            Zr = np.asarray(Z[rep][:, cols], dtype=np.float32)
            Wk = W[cols]
            base_sel.append(float(f1_score(yte[sel], np.argmax(class_sums(Zs, Wk, cpc), 1), average="macro")))
            base_rep.append(float(f1_score(yte[rep], np.argmax(class_sums(Zr, Wk, cpc), 1), average="macro")))
            for Bv in B_SNAPS:
                pis = [np.array(snaps[(seed, pool, i)][Bv][0]) for i in range(NCLS)]
                cfs = [np.array(snaps[(seed, pool, i)][Bv][1]) for i in range(NCLS)]
                for thr in PI_THRS:
                    fs, as_, sp, sa = score(Zs, yte[sel], pis, cfs, thr)
                    fr, ar, spr, sar = score(Zr, yte[rep], pis, cfs, thr)
                    grid.setdefault((Bv, thr), []).append(
                        {"seed": seed, "sel_f1": fs, "sel_acc": as_,
                         "rep_f1": fr, "rep_acc": ar, "S_pos": sp, "S": sa})
        gm = {f"{Bv}|{thr}": {"sel_f1": float(np.mean([r["sel_f1"] for r in v])),
                              "rep_f1": float(np.mean([r["rep_f1"] for r in v])),
                              "rep_f1_sd": float(np.std([r["rep_f1"] for r in v])),
                              "rep_acc": float(np.mean([r["rep_acc"] for r in v])),
                              "rep_acc_sd": float(np.std([r["rep_acc"] for r in v])),
                              "mean_S_pos": float(np.mean([np.mean(r["S_pos"]) for r in v])),
                              "per_seed": v}
              for (Bv, thr), v in grid.items()}
        best = max(gm, key=lambda k: gm[k]["sel_f1"])
        Bb, tb = best.split("|")
        out["pools"][pool] = {
            "clauses_per_class": cpc, "pool": cpc * NCLS,
            "base_sel_f1": float(np.mean(base_sel)),
            "base_rep_f1": float(np.mean(base_rep)),
            "base_rep_f1_sd": float(np.std(base_rep)),
            "selected_B": int(Bb), "selected_pi_thr": float(tb),
            "reported": gm[best], "grid": gm}
        print(f"[{pool}] selected B={Bb} pi_thr={tb} on sel "
              f"(sel {gm[best]['sel_f1']:.4f}) -> REPORT rep_f1="
              f"{gm[best]['rep_f1']:.4f}+-{gm[best]['rep_f1_sd']:.4f}, "
              f"base_rep={np.mean(base_rep):.4f}", flush=True)

    with open(f"{BASE}/threeway_binary_pruned.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[done] results/threeway_binary_pruned.json", flush=True)


if __name__ == "__main__":
    main()
