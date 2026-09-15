"""
Leak-free three-way protocol for the binary CAS arm (Table III).

Same permutation and split as threeway_cas.py: fit half / selection quarter /
report quarter of the 80,868-row test set. CPSS runs to B=25 with snapshots
at 5/10/15/20/25; (B, pi_thr) are chosen on the selection quarter by macro-F1
and the report quarter is read once.

Includes a reproduction check: the old protocol (fit half -> eval half,
B=15, pi_thr=0.6, eval = sel+rep combined) must land within noise of the
published 0.928 attack-F1 / 0.940 AUROC before the clean numbers are trusted.

Output: results/threeway_binary.json
"""
import json, os, pickle, time
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NCLS, NC, T = 2, 400, 320
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25


def splits(n):
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def prep():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xte = d["Xte"]
    yb = (d["yte"] != 0).astype(np.uint8)
    for seed in SEEDS:
        zp = f"{BASE}/Zte_binary_seed{seed}.npy"
        if os.path.exists(zp):
            continue
        with open(f"{BASE}/binary_tm_model_seed{seed}.pkl", "rb") as f:
            tm = pickle.load(f)
        np.save(zp, tm.transform(Xte).astype(np.uint8))
        W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                            for c in range(NCLS)])
        np.save(f"{BASE}/W_binary_seed{seed}.npy", W)
        print(f"[prep] seed {seed} cached", flush=True)
    return yb


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


def score_matrix(Zev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], NCLS))
    for i in range(NCLS):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    return S


def metrics(yev, S):
    pred = np.argmax(S, 1)
    return {
        "macro_f1": float(f1_score(yev, pred, average="macro")),
        "attack_f1": float(f1_score(yev, pred, pos_label=1)),
        "attack_recall": float(recall_score(yev, pred, pos_label=1)),
        "attack_precision": float(precision_score(yev, pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(yev, S[:, 1])),
    }


def tm_scores(Z, W):
    cs = np.column_stack([Z[:, c * NC:(c + 1) * NC] @ W[c * NC:(c + 1) * NC]
                          for c in range(NCLS)])
    S = np.zeros_like(cs)
    S[:, 0] = cs[:, 0]
    S[:, 1] = np.clip(cs[:, 1], 0, T) / T  # margin used for AUROC
    return cs, S


def job(args):
    seed, ci = args
    t0 = time.time()
    yb = (np.load(f"{BASE}/booleanized_data.npz")["yte"] != 0).astype(np.uint8)
    Z = np.load(f"{BASE}/Zte_binary_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    Zf = np.asarray(Z[fit], dtype=np.float32)
    ybin = (yb[fit] == ci).astype(int)
    out = cpss_snap(Zf, ybin, seed=seed + ci)
    print(f"[cpss] seed{seed} class{ci} {time.time()-t0:.0f}s", flush=True)
    return (seed, ci, out)


def main():
    yb = prep()
    n = len(yb)
    fit, sel, rep = splits(n)
    print(f"[split] fit={len(fit)} sel={len(sel)} rep={len(rep)} "
          f"attack prev rep={yb[rep].mean():.3f}", flush=True)

    jobs = [(s, c) for s in SEEDS for c in range(NCLS)]
    with Pool(processes=6) as p:
        res = p.map(job, jobs)
    snaps = {(s, c): o for s, c, o in res}

    out = {"n": {"fit": len(fit), "sel": len(sel), "rep": len(rep)},
           "pi_thrs": PI_THRS, "b_snaps": list(B_SNAPS), "seeds": SEEDS}

    # ---------- reproduction check: old protocol, B=15, pi_thr=0.6 ----------
    repro = []
    for s in SEEDS:
        Z = np.asarray(np.load(f"{BASE}/Zte_binary_seed{s}.npy"), dtype=np.float32)
        pis = [np.array(snaps[(s, c)][15][0]) for c in range(NCLS)]
        cfs = [np.array(snaps[(s, c)][15][1]) for c in range(NCLS)]
        ev = np.concatenate([sel, rep])  # = old eval half
        m = metrics(yb[ev], score_matrix(Z[ev], pis, cfs, 0.6))
        repro.append(m)
    out["repro_old_protocol_B15_thr0.6"] = repro
    print("[repro] attack F1 "
          f"{np.mean([m['attack_f1'] for m in repro]):.4f}  AUROC "
          f"{np.mean([m['auroc'] for m in repro]):.4f}  (published 0.928 / 0.940)",
          flush=True)

    # ---------- clean protocol ----------
    grid = {}
    tm_rep, tm_sel = [], []
    for s in SEEDS:
        Z = np.asarray(np.load(f"{BASE}/Zte_binary_seed{s}.npy"), dtype=np.float32)
        W = np.load(f"{BASE}/W_binary_seed{s}.npy")
        for idx, store in ((sel, tm_sel), (rep, tm_rep)):
            cs, S = tm_scores(Z[idx], W)
            m = metrics(yb[idx], S)
            m["macro_f1"] = float(f1_score(yb[idx], np.argmax(cs, 1), average="macro"))
            m["attack_f1"] = float(f1_score(yb[idx], np.argmax(cs, 1), pos_label=1))
            m["attack_recall"] = float(recall_score(yb[idx], np.argmax(cs, 1), pos_label=1))
            m["attack_precision"] = float(precision_score(yb[idx], np.argmax(cs, 1), pos_label=1))
            store.append(m)
        for Bv in B_SNAPS:
            pis = [np.array(snaps[(s, c)][Bv][0]) for c in range(NCLS)]
            cfs = [np.array(snaps[(s, c)][Bv][1]) for c in range(NCLS)]
            for thr in PI_THRS:
                ms = metrics(yb[sel], score_matrix(Z[sel], pis, cfs, thr))
                mr = metrics(yb[rep], score_matrix(Z[rep], pis, cfs, thr))
                grid.setdefault((Bv, thr), []).append({"seed": s, "sel": ms, "rep": mr})

    agg = {f"{Bv}|{thr}": {
        "sel_macro_f1": float(np.mean([r["sel"]["macro_f1"] for r in v])),
        "rep": {k: float(np.mean([r["rep"][k] for r in v]))
                for k in ("macro_f1", "attack_f1", "attack_recall",
                          "attack_precision", "auroc")},
        "rep_sd": {k: float(np.std([r["rep"][k] for r in v]))
                   for k in ("macro_f1", "attack_f1", "auroc")},
        "per_seed": v} for (Bv, thr), v in grid.items()}
    best = max(agg, key=lambda k: agg[k]["sel_macro_f1"])
    Bb, tb = best.split("|")
    out["tm_baseline"] = {
        "sel": {k: float(np.mean([m[k] for m in tm_sel])) for k in tm_sel[0]},
        "rep": {k: float(np.mean([m[k] for m in tm_rep])) for k in tm_rep[0]},
        "rep_sd": {k: float(np.std([m[k] for m in tm_rep])) for k in tm_rep[0]}}
    out["selected_B"] = int(Bb)
    out["selected_pi_thr"] = float(tb)
    out["reported"] = agg[best]
    out["grid"] = agg
    print(f"[binary] selected B={Bb} pi_thr={tb} on sel "
          f"(macro-F1 {agg[best]['sel_macro_f1']:.4f}) -> REPORT "
          f"attackF1={agg[best]['rep']['attack_f1']:.4f} "
          f"R={agg[best]['rep']['attack_recall']:.4f} "
          f"P={agg[best]['rep']['attack_precision']:.4f} "
          f"AUROC={agg[best]['rep']['auroc']:.4f}", flush=True)
    print(f"[binary] TM base REPORT attackF1={out['tm_baseline']['rep']['attack_f1']:.4f} "
          f"AUROC={out['tm_baseline']['rep']['auroc']:.4f}", flush=True)

    with open(f"{BASE}/threeway_binary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[done] results/threeway_binary.json", flush=True)


if __name__ == "__main__":
    main()
