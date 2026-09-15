"""
Leak-free CAS protocol: CPSS is fitted on held-out TRAIN data, never on test.

The published three-way protocol splits the TEST set into fit/sel/rep and
fits CPSS on the fit half. Because the 22 continuous features are thermometer
-encoded into 10 quantile bins, the 80,868-row test set collapses to 3,592
distinct (features,label) patterns, so 97.3% of report-quarter rows have an
exact twin in the fit half. Anything fitted on `fit` therefore memorises
`rep`. Deduplicating raw rows does not help: the dedup test set collapses to
11,393 patterns and the twin rate rises to 99.1%.

Here the TRAIN set is split three ways instead and the test set is untouched
until the single final report:

    core = 70%  TM training          (the TM is retrained on this)
    cpss = 20%  CPSS half-fits       (signature extraction)
    selm = 10%  operating point      (k*, B, pi_thr, baseline hyperparams)
    TEST = 100% reported once

The TM is retrained on `core` so that `cpss` is genuinely unseen by the TM
whose clauses CPSS selects over. Baselines train on `core` and pick
hyperparameters on `selm`, giving every method identical information access.

    python3 scripts/heldout_protocol.py --stage train
    python3 scripts/heldout_protocol.py --stage cas
    python3 scripts/heldout_protocol.py --stage baselines
"""
import argparse, json, os, pickle, time
import numpy as np
from multiprocessing import Pool, Process
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE = "/FPTM/CAS_IoMT_Empirical/results"
OUT = f"{BASE}/heldout"
NPZ = "booleanized_data.npz"
DATA_DIR = BASE      # results/ for CICIoMT2024, results_medsec/ for MedSec-25
CPSS_CAP = 0          # 0 = use the whole cpss split; >0 = stratified cap per class
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
NC, NCLS = 400, 6
TM_HP = dict(number_of_clauses=NC, T=320, s=8.0, epochs=25)
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25
EPS = 0.01
STEP = 5
FRACS = (0.70, 0.20, 0.10)


def train_split(y, seed=0):
    """Stratified core/cpss/selm split of the TRAINING set."""
    rng = np.random.RandomState(seed)
    core, cpss, selm = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(FRACS[0] * n), int((FRACS[0] + FRACS[1]) * n)
        core.append(idx[:a]); cpss.append(idx[a:b]); selm.append(idx[b:])
    f = lambda L: np.sort(np.concatenate(L))
    return f(core), f(cpss), f(selm)


def load():
    d = np.load(f"{DATA_DIR}/{NPZ}")
    return d["Xtr"], d["ytr"].astype(int), d["Xte"], d["yte"].astype(int)


def cap_per_class(idx, y, cap, seed=0):
    """Stratified subsample of `idx`; L1 CPSS on the full split is intractable
    at 1.6M-row scale, and a per-class cap is ample for stability selection."""
    if not cap:
        return idx
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        keep.append(ci if len(ci) <= cap else rng.choice(ci, cap, replace=False))
    return np.sort(np.concatenate(keep))


# ------------------------------------------------------------------ stage: train
def stage_train():
    os.makedirs(OUT, exist_ok=True)
    from tmu.models.classification.vanilla_classifier import TMClassifier
    Xtr, ytr, Xte, yte = load()
    core, cpss, selm = train_split(ytr)
    json.dump({"core": len(core), "cpss": len(cpss), "selm": len(selm),
               "test": len(yte)}, open(f"{OUT}/split_sizes.json", "w"), indent=2)
    print(f"[split] core={len(core)} cpss={len(cpss)} selm={len(selm)} test={len(yte)}",
          flush=True)
    def _one(seed):
        p = f"{OUT}/tm_core_seed{seed}.pkl"
        if os.path.exists(p):
            print(f"[train] seed {seed} exists, skip", flush=True); return
        tm = TMClassifier(number_of_clauses=NC, T=TM_HP["T"], s=TM_HP["s"],
                          weighted_clauses=True, platform="CPU", seed=seed)
        t0 = time.time()
        for ep in range(TM_HP["epochs"]):
            tm.fit(Xtr[core], ytr[core].astype(np.uint32))
        f1 = f1_score(yte, tm.predict(Xte), average="macro")
        print(f"[train] seed {seed} {time.time()-t0:.0f}s  TEST macro-F1={f1:.4f}", flush=True)
        with open(p, "wb") as f:
            pickle.dump(tm, f)

    procs = [Process(target=_one, args=(s_,)) for s_ in SEEDS]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def cache_Z(seed, Xtr, cpss, selm, Xte):
    """Clause-activation matrices for the three evaluation surfaces."""
    paths = {k: f"{OUT}/Z_{k}_seed{seed}.npy" for k in ("cpss", "selm", "test")}
    if all(os.path.exists(v) for v in paths.values()):
        return paths
    with open(f"{OUT}/tm_core_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    for k, X in (("cpss", Xtr[cpss]), ("selm", Xtr[selm]), ("test", Xte)):
        if not os.path.exists(paths[k]):
            np.save(paths[k], tm.transform(X).astype(np.uint8))
    W = np.concatenate([np.asarray(tm.weight_banks[c].get_weights(), np.float32)
                        for c in range(NCLS)])
    np.save(f"{OUT}/W_seed{seed}.npy", W)
    print(f"[cache] seed {seed} done", flush=True)
    return paths


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
    ng = len(KAPPA_GRID); out = {}
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


def score(Zev, yev, pis, cfs, thr):
    S = np.zeros((Zev.shape[0], NCLS)); sizes = []
    for i in range(NCLS):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        sizes.append(len(Sp))
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Zev[:, Sp] @ pi[Sp] - (Zev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    pred = np.argmax(S, 1)
    return (float(f1_score(yev, pred, average="macro")),
            float((pred == yev).mean()), sizes)


def _job(a):
    seed, pool, fi, cols, ycp = a
    t0 = time.time()
    Z = np.load(f"{OUT}/Z_cpss_seed{seed}.npy", mmap_mode="r")
    Zf = np.asarray(Z[:, cols], dtype=np.float32)
    out = cpss_snap(Zf, (ycp == fi).astype(int), seed + fi)
    print(f"[cpss] seed{seed} {pool} {FAMILIES[fi]} {time.time()-t0:.0f}s", flush=True)
    return (seed, pool, fi, out)


# -------------------------------------------------------------------- stage: cas
def stage_cas():
    Xtr, ytr, Xte, yte = load()
    core, cpss, selm = train_split(ytr)
    cpss = cap_per_class(cpss, ytr, CPSS_CAP)
    print(f"[cas] cpss rows used for CPSS: {len(cpss)}", flush=True)
    ycp, ysl = ytr[cpss], ytr[selm]
    for seed in SEEDS:
        cache_Z(seed, Xtr, cpss, selm, Xte)

    # ---- pruning budget k*, chosen on selm (never on test) ----
    ks = list(range(0, NC, STEP))
    curves = []
    for seed in SEEDS:
        Zs = np.asarray(np.load(f"{OUT}/Z_selm_seed{seed}.npy"), dtype=np.float32)
        W = np.load(f"{OUT}/W_seed{seed}.npy")
        aw = np.abs(W)
        order = {c: c * NC + np.argsort(aw[c * NC:(c + 1) * NC], kind="stable")
                 for c in range(NCLS)}
        cs = []
        for k in ks:
            Wm = W.copy()
            if k:
                Wm[np.concatenate([order[c][:k] for c in range(NCLS)])] = 0.0
            cs.append(float(f1_score(ysl, np.argmax(class_sums(Zs, Wm, NC), 1),
                                     average="macro")))
        curves.append(cs)
        print(f"[prune] seed {seed} swept", flush=True)
    msel = np.array(curves).mean(0)
    ok = [i for i, v in enumerate(msel) if v >= msel[0] - EPS]
    kstar = ks[max(ok)]; keep = NC - kstar
    print(f"[prune] k*={kstar} -> keep {keep}/class (pool {keep*NCLS})", flush=True)

    pools = {}
    for seed in SEEDS:
        W = np.load(f"{OUT}/W_seed{seed}.npy")
        pools[(seed, "full")] = np.arange(NC * NCLS)
        pools[(seed, "pruned")] = keep_cols(W, keep)
    jobs = [(s, p, fi, c, ycp) for (s, p), c in pools.items() for fi in range(NCLS)]
    with Pool(processes=min(36, len(jobs))) as p:
        res = p.map(_job, jobs)
    snaps = {(s, po, fi): o for s, po, fi, o in res}

    out = {"kstar_removed_per_class": kstar, "keep_per_class": keep,
           "n": {"core": len(core), "cpss": len(cpss), "selm": len(selm),
                 "test": len(yte)},
           "prune_curve_selm": curves, "ks": ks, "pools": {}}
    for pool in ("full", "pruned"):
        cpc = NC if pool == "full" else keep
        base_test, grid = [], {}
        for seed in SEEDS:
            cols = pools[(seed, pool)]
            W = np.load(f"{OUT}/W_seed{seed}.npy")[cols]
            Zs = np.asarray(np.load(f"{OUT}/Z_selm_seed{seed}.npy")[:, cols], np.float32)
            Zt = np.asarray(np.load(f"{OUT}/Z_test_seed{seed}.npy")[:, cols], np.float32)
            bt = float(f1_score(yte, np.argmax(class_sums(Zt, W, cpc), 1), average="macro"))
            ba = float((np.argmax(class_sums(Zt, W, cpc), 1) == yte).mean())
            base_test.append({"seed": seed, "test_f1": bt, "test_acc": ba})
            for Bv in B_SNAPS:
                pis = [np.array(snaps[(seed, pool, i)][Bv][0]) for i in range(NCLS)]
                cfs = [np.array(snaps[(seed, pool, i)][Bv][1]) for i in range(NCLS)]
                for thr in PI_THRS:
                    fs, _, _ = score(Zs, ysl, pis, cfs, thr)
                    ft, at, sz = score(Zt, yte, pis, cfs, thr)
                    grid.setdefault((Bv, thr), []).append(
                        {"seed": seed, "selm_f1": fs, "test_f1": ft,
                         "test_acc": at, "S_pos": sz})
        gm = {f"{B}|{t}": {"selm_f1": float(np.mean([r["selm_f1"] for r in v])),
                           "test_f1": float(np.mean([r["test_f1"] for r in v])),
                           "test_f1_sd": float(np.std([r["test_f1"] for r in v])),
                           "test_acc": float(np.mean([r["test_acc"] for r in v])),
                           "test_acc_sd": float(np.std([r["test_acc"] for r in v])),
                           "mean_S_pos": float(np.mean([np.mean(r["S_pos"]) for r in v])),
                           "per_seed": v}
              for (B, t), v in grid.items()}
        best = max(gm, key=lambda k: gm[k]["selm_f1"])
        out["pools"][pool] = {
            "tm_base": {"test_f1_mean": float(np.mean([r["test_f1"] for r in base_test])),
                        "test_f1_sd": float(np.std([r["test_f1"] for r in base_test])),
                        "test_acc_mean": float(np.mean([r["test_acc"] for r in base_test])),
                        "test_acc_sd": float(np.std([r["test_acc"] for r in base_test])),
                        "per_seed": base_test},
            "selected": best, "cas": gm[best], "grid": gm}
        print(f"[cas] {pool}: TM {out['pools'][pool]['tm_base']['test_f1_mean']:.4f} | "
              f"CAS@{best} {gm[best]['test_f1']:.4f}+-{gm[best]['test_f1_sd']:.4f}", flush=True)
    json.dump(out, open(f"{OUT}/heldout_cas.json", "w"), indent=2)
    print(f"[out] {OUT}/heldout_cas.json", flush=True)


# -------------------------------------------------------------- stage: baselines
def load_raw_features():
    """The 22 continuous features, in the same row order as the booleanized
    arrays; the label sequence is asserted to match before use."""
    from data_prep import FEATURE_COLS, LABEL_COL, FAMILIES, load_and_sample, clean_inf
    tr, te = load_and_sample()
    tr, te = clean_inf(tr, te)
    f = lambda d: (d[FEATURE_COLS].to_numpy(np.float64).astype(np.float32),
                   d[LABEL_COL].map({x: i for i, x in enumerate(FAMILIES)}).to_numpy(int))
    return f(tr), f(te)


def stage_baselines(models, train_on="core", features="bool"):
    """train_on='core+cpss' matches the total row count CAS fits on across its
    two stages (TM training + CPSS extraction), so the baselines are not given
    22% less data than the method they are compared against."""
    import nontm_baselines as nb
    Xtr, ytr, Xte, yte = load()
    if features == "raw":
        (Xr_tr, yr_tr), (Xr_te, yr_te) = load_raw_features()
        assert np.array_equal(yr_tr, ytr), "raw train row order does not match"
        assert np.array_equal(yr_te, yte), "raw test row order does not match"
        Xtr, Xte = Xr_tr, Xr_te
        print("[align] raw features verified against booleanized labels", flush=True)
    core, cpss, selm = train_split(ytr)
    idx = np.sort(np.concatenate([core, cpss])) if train_on == "core+cpss" else core
    print(f"[data] baselines train_on={train_on} rows={len(idx)}", flush=True)
    Xc, yc = Xtr[idx].astype(np.float32), ytr[idx]
    Xs, ys = Xtr[selm].astype(np.float32), ytr[selm]
    Xt = Xte.astype(np.float32)
    res = {}
    for name in models:
        fn, grid = nb.GRIDS[name]
        per_seed = []
        for seed in SEEDS:
            best, bhp, bm = -1.0, None, None
            for hp in grid:
                t0 = time.time()
                m = fn(Xc, yc, seed, hp)
                p = m.predict(Xs)
                f1 = float(f1_score(ys, p, average="macro"))
                print(f"  [{name}] seed{seed} hp={hp} selm_f1={f1:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                if f1 > best:
                    best, bhp, bm = f1, hp, m
            pt = bm.predict(Xt)
            f1t = float(f1_score(yte, pt, average="macro"))
            act = float((pt == yte).mean())
            per_seed.append({"seed": seed, "hp": str(bhp), "selm_f1": best,
                             "test_f1": f1t, "test_acc": act})
            print(f"  [{name}] seed{seed} CHOSE {bhp} -> TEST f1={f1t:.4f} acc={act:.4f}",
                  flush=True)
        f = [r["test_f1"] for r in per_seed]; a = [r["test_acc"] for r in per_seed]
        res[name] = {"model": name,
                     "test_f1_mean": float(np.mean(f)), "test_f1_sd": float(np.std(f)),
                     "test_acc_mean": float(np.mean(a)), "test_acc_sd": float(np.std(a)),
                     "per_seed": per_seed}
        print(f"[done] {name} TEST f1={res[name]['test_f1_mean']:.4f}"
              f"+-{res[name]['test_f1_sd']:.4f}", flush=True)
        tag = ("" if train_on == "core" else "_coreplus") + \
              ("" if features == "bool" else "_raw")
        json.dump(res, open(f"{OUT}/heldout_baselines{tag}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "cas", "baselines"])
    ap.add_argument("--models", default="dt,l1,rf,xgb")
    ap.add_argument("--train-on", default="core", choices=["core", "core+cpss"])
    ap.add_argument("--npz", default="booleanized_data.npz")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cpss-cap", type=int, default=0)
    ap.add_argument("--features", default="bool", choices=["bool", "raw"],
                    help="bool: 154 thermometer bits. raw: the 22 continuous features.")
    ap.add_argument("--data-dir", default=None,
                    help="results (CICIoMT2024) or results_medsec (MedSec-25)")
    a = ap.parse_args()
    global NPZ, OUT, CPSS_CAP, DATA_DIR, NCLS, FAMILIES
    NPZ, CPSS_CAP = a.npz, a.cpss_cap
    if a.data_dir:
        DATA_DIR = f"/FPTM/CAS_IoMT_Empirical/{a.data_dir}"
    # class count comes from the data, not a hard-coded constant
    _y = np.load(f"{DATA_DIR}/{NPZ}")["ytr"]
    NCLS = int(np.unique(_y).size)
    if NCLS != len(FAMILIES):
        FAMILIES = [f"class{i}" for i in range(NCLS)]
        try:
            import json as _j
            _n = _j.load(open(f"{DATA_DIR}/literal_names.json"))
            if isinstance(_n, dict) and len(_n.get("families", [])) == NCLS:
                FAMILIES = _n["families"]
        except Exception:
            pass
    if a.outdir:
        OUT = f"{BASE}/{a.outdir}"
    os.makedirs(OUT, exist_ok=True)
    print(f"[cfg] data={DATA_DIR}/{NPZ} out={OUT} cpss_cap={CPSS_CAP} "
          f"n_classes={NCLS} families={FAMILIES}", flush=True)
    if a.stage == "train":
        stage_train()
    elif a.stage == "cas":
        stage_cas()
    else:
        stage_baselines([m.strip() for m in a.models.split(",")], a.train_on,
                        a.features)


if __name__ == "__main__":
    main()
