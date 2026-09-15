"""
Retrain the binary TM at the grid-search winner and re-run the binary CAS
comparison against a *calibrated* baseline.

The paper's binary baseline is TM argmax, which is uncalibrated under the
90/10 training prior and the Benign covariate shift documented in
binary_cas_diagnosis.py. This script reports three baselines on the report
quarter -- TM argmax, TM with one threshold tuned on the selection quarter,
and CAS -- so the part of the CAS gain attributable to signature selection
is separated from the part attributable to calibration.

Usage: python3 scripts/binary_retune_rerun.py [--config-rank 0]
Output: results/binary_retuned.json, results/binary_tm_retuned_seed{S}.pkl
"""
import argparse, json, os, pickle, time
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25
TAG = ""


def splits(n):
    p = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return p[:h], p[h:h + q], p[h + q:]


def cpss_snap(Z, ybin, seed):
    n, m = Z.shape
    rng = np.random.RandomState(seed)
    half = n // 2
    sel = np.zeros(m, np.int64); csum = np.zeros(m, np.float64)
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
            out[b + 1] = ((sel / (2 * (b + 1))).copy(),
                          (csum / (2 * (b + 1) * len(KAPPA_GRID))).copy())
    return out


def score_matrix(Zev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], 2))
    for i in range(2):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    return S


def metrics(y, S):
    p = np.argmax(S, 1)
    return {"macro_f1": float(f1_score(y, p, average="macro")),
            "attack_f1": float(f1_score(y, p, pos_label=1)),
            "attack_recall": float(recall_score(y, p, pos_label=1)),
            "attack_precision": float(precision_score(y, p, pos_label=1, zero_division=0)),
            "benign_recall": float(recall_score(y, p, pos_label=0)),
            "auroc": float(roc_auc_score(y, S[:, 1] - S[:, 0]))}


def train_one(args):
    seed, cfg = args
    from tmu.models.classification.vanilla_classifier import TMClassifier
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xtr, ytr = d["Xtr"], (d["ytr"] != 0).astype(np.uint32)
    if cfg["balance"] == "oversample":
        rng = np.random.RandomState(seed)
        mino = np.where(ytr == 0)[0]
        extra = rng.choice(mino, int((ytr == 1).sum()) - len(mino), replace=True)
        idx = rng.permutation(np.concatenate([np.arange(len(ytr)), extra]))
        Xtr, ytr = np.ascontiguousarray(Xtr[idx]), np.ascontiguousarray(ytr[idx])
    if os.path.exists(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy") and \
       os.path.exists(f"{BASE}/W_binary_retuned{TAG}_seed{seed}.npy"):
        print(f"[train] seed {seed} cached, skipping", flush=True)
        return seed
    tm = TMClassifier(number_of_clauses=cfg["clauses"], T=cfg["T"], s=cfg["s"],
                      weighted_clauses=True, platform="CPU", seed=seed)
    t0 = time.time()
    yb_all = (d["yte"] != 0).astype(int)
    _, sel_idx, _ = splits(len(yb_all))
    Xsel, ysel = d["Xte"][sel_idx], yb_all[sel_idx]
    curve = []
    for ep in range(1, cfg["epochs"] + 1):
        tm.fit(Xtr, ytr)
        if ep in (1, 2, 3, 5, 7, 10, 15, 20, 25) or ep == cfg["epochs"]:
            f = float(f1_score(ysel, tm.predict(Xsel), average="macro"))
            curve.append({"epoch": ep, "sel_macro_f1": f})
            print(f"[ckpt] seed {seed} ep{ep} sel macro-F1 {f:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(f"{BASE}/tm_curve{TAG}_seed{seed}.json", "w") as f:
        json.dump(curve, f)
    with open(f"{BASE}/binary_tm_retuned{TAG}_seed{seed}.pkl", "wb") as f:
        pickle.dump(tm, f)
    Z = tm.transform(d["Xte"]).astype(np.uint8)
    np.save(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy", Z)
    W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                        for c in range(2)])
    np.save(f"{BASE}/W_binary_retuned{TAG}_seed{seed}.npy", W)
    print(f"[train] seed {seed} done in {time.time()-t0:.0f}s", flush=True)
    return seed


def cpss_job(args):
    seed, ci, nc = args
    d = np.load(f"{BASE}/booleanized_data.npz")
    yb = (d["yte"] != 0).astype(np.uint8)
    Z = np.load(f"{BASE}/Zte_binary_retuned{TAG}_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    out = cpss_snap(np.asarray(Z[fit], np.float32), (yb[fit] == ci).astype(int), seed + ci)
    print(f"[cpss] seed{seed} class{ci} done", flush=True)
    return (seed, ci, {k: (v[0].tolist(), v[1].tolist()) for k, v in out.items()})


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-rank", type=int, default=0,
                    help="rank in the grid ordered by the chosen selection statistic")
    ap.add_argument("--select", choices=["max-epoch", "mean-epoch"], default="mean-epoch",
                    help="max-epoch takes the best checkpoint of each config, which is a "
                         "maximum over 9 noisy reads and is optimistically biased; "
                         "mean-epoch averages the checkpoints and picks the median epoch")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--explicit", default="",
                    help="bypass the grid: 'clauses,T,s,balance,epochs' "
                         "e.g. '1000,750,3.0,oversample,15'")
    a = ap.parse_args()
    TAG = a.tag
    if a.explicit:
        c, T, sv, bal, ep = a.explicit.split(",")
        cfg = {"clauses": int(c), "T": int(T), "s": float(sv),
               "balance": bal.strip(), "epochs": int(ep)}
        nc = cfg["clauses"]
        print(f"[cfg] {cfg} (explicit, not grid-selected)", flush=True)
        return run_pipeline(cfg, nc, a, grid_stat=None, select_rule="explicit")
    g = json.load(open(f"{BASE}/binary_tm_gridsearch.json"))["grid"]
    if a.select == "max-epoch":
        g.sort(key=lambda r: -r["best_by_sel"]["sel_macro_f1"])
        pick = lambda r: r["best_by_sel"]["epoch"]
        stat = lambda r: r["best_by_sel"]["sel_macro_f1"]
    else:
        mean = lambda r: float(np.mean([c["sel_macro_f1"] for c in r["curve"]]))
        g.sort(key=lambda r: -mean(r))
        # epoch whose sel score is closest to the config's own epoch-mean
        pick = lambda r: min(r["curve"], key=lambda c: abs(c["sel_macro_f1"] - mean(r)))["epoch"]
        stat = mean
    r = g[a.config_rank]
    cfg = {"clauses": r["clauses"], "T": r["T"], "s": r["s"],
           "balance": r["balance"], "epochs": pick(r)}
    nc = cfg["clauses"]
    print(f"[cfg] {cfg} (grid {a.select} sel macro-F1 {stat(r):.4f})", flush=True)
    return run_pipeline(cfg, nc, a, grid_stat=stat(r), select_rule=a.select)


def run_pipeline(cfg, nc, a, grid_stat, select_rule):
    with Pool(3) as p:
        p.map(train_one, [(s, cfg) for s in SEEDS])
    with Pool(6) as p:
        res = p.map(cpss_job, [(s, c, nc) for s in SEEDS for c in range(2)])
    snaps = {(s, c): {int(k): (np.array(v[0]), np.array(v[1])) for k, v in o.items()}
             for s, c, o in res}

    yb = (np.load(f"{BASE}/booleanized_data.npz")["yte"] != 0).astype(int)
    fit, sel, rep = splits(len(yb))
    per_seed, grid = [], {}
    for s in SEEDS:
        Z = np.asarray(np.load(f"{BASE}/Zte_binary_retuned{TAG}_seed{s}.npy"), np.float32)
        W = np.load(f"{BASE}/W_binary_retuned{TAG}_seed{s}.npy")
        cs = np.column_stack([Z[:, c * nc:(c + 1) * nc] @ W[c * nc:(c + 1) * nc]
                              for c in range(2)])
        m_sel, m_rep = cs[sel, 1] - cs[sel, 0], cs[rep, 1] - cs[rep, 0]
        cand = np.quantile(m_sel, np.linspace(0.001, 0.999, 400))
        fs = [f1_score(yb[sel], (m_sel > t).astype(int), average="macro") for t in cand]
        t = float(cand[int(np.argmax(fs))])
        per_seed.append({
            "seed": s,
            "tm_argmax_rep": metrics(yb[rep], cs[rep]),
            "tm_tuned_threshold": {
                "threshold": t,
                "sel_macro_f1": float(max(fs)),
                "rep_macro_f1": float(f1_score(yb[rep], (m_rep > t).astype(int), average="macro")),
                "rep_benign_recall": float(recall_score(yb[rep], (m_rep > t).astype(int), pos_label=0)),
                "rep_auroc": float(roc_auc_score(yb[rep], m_rep))}})
        for Bv in B_SNAPS:
            pis = [snaps[(s, c)][Bv][0] for c in range(2)]
            cfs = [snaps[(s, c)][Bv][1] for c in range(2)]
            for thr in PI_THRS:
                grid.setdefault((Bv, thr), []).append(
                    {"seed": s,
                     "sel": metrics(yb[sel], score_matrix(Z[sel], pis, cfs, thr)),
                     "rep": metrics(yb[rep], score_matrix(Z[rep], pis, cfs, thr))})

    agg = {f"{B}|{th}": {
        "sel_macro_f1": float(np.mean([x["sel"]["macro_f1"] for x in v])),
        "rep": {k: float(np.mean([x["rep"][k] for x in v])) for k in v[0]["rep"]},
        "rep_sd": {k: float(np.std([x["rep"][k] for x in v])) for k in v[0]["rep"]},
        "per_seed": v} for (B, th), v in grid.items()}
    best = max(agg, key=lambda k: agg[k]["sel_macro_f1"])
    out = {"config": cfg, "grid_sel_macro_f1": grid_stat, "grid_select_rule": select_rule,
           "tm_epoch_curve_sel": {s: json.load(open(f"{BASE}/tm_curve{TAG}_seed{s}.json"))
                                  for s in SEEDS
                                  if os.path.exists(f"{BASE}/tm_curve{TAG}_seed{s}.json")},
           "selected_B_pithr": best, "cas_reported": agg[best], "cas_grid": agg,
           "baselines_per_seed": per_seed,
           "summary": {
               "tm_argmax": float(np.mean([p["tm_argmax_rep"]["macro_f1"] for p in per_seed])),
               "tm_argmax_sd": float(np.std([p["tm_argmax_rep"]["macro_f1"] for p in per_seed])),
               "tm_tuned": float(np.mean([p["tm_tuned_threshold"]["rep_macro_f1"] for p in per_seed])),
               "tm_tuned_sd": float(np.std([p["tm_tuned_threshold"]["rep_macro_f1"] for p in per_seed])),
               "cas": agg[best]["rep"]["macro_f1"], "cas_sd": agg[best]["rep_sd"]["macro_f1"]}}
    with open(f"{BASE}/binary_retuned{TAG}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2), flush=True)
    print(f"[done] results/binary_retuned{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
