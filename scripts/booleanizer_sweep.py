"""
Is the thermometer booleanizer capping the TM's macro-F1?

The 10-bin quantile thermometer maps 148k distinct raw rows onto 4,901 distinct
154-bit codes, so the encoding -- not the classifier -- may be the binding
constraint. This sweeps bin count and binning strategy and reports, for each:

  bits          width of the Boolean input
  patterns      distinct (features,label) codes in train / test
  ceiling       accuracy of the accuracy-optimal function of those bits,
                measured in-sample on test (a valid upper bound for ANY model)
  dt / rf       fast proxies trained on `core`, scored on the full test set

Stage 2 runs the TM itself at the most promising settings, since a TM costs
~260s/seed and the proxies cost seconds.

    python3 scripts/booleanizer_sweep.py --stage scan
    python3 scripts/booleanizer_sweep.py --stage tm --configs 10:quantile,30:quantile
"""
import argparse, json, sys, time
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from data_prep import FEATURE_COLS, LABEL_COL, FAMILIES
from heldout_protocol import train_split, OUT, SEEDS

BINS = [3, 5, 10, 15, 20, 30]
STRATS = ["quantile", "uniform"]


def raw_capped():
    from data_prep import load_and_sample, clean_inf
    tr, te = load_and_sample()
    tr, te = clean_inf(tr, te)
    f = lambda d: (d[FEATURE_COLS].to_numpy(np.float64),
                   d[LABEL_COL].map({x: i for i, x in enumerate(FAMILIES)}).to_numpy(int))
    return f(tr), f(te)


def booleanize(Xtr, Xte, n_bins, strategy):
    """Per-feature quantile/uniform thermometer, edges fit on train only."""
    btr, bte = [], []
    for j in range(Xtr.shape[1]):
        ctr, cte = Xtr[:, j:j + 1], Xte[:, j:j + 1]
        nu = len(np.unique(ctr))
        nb = min(n_bins, max(2, nu))
        try:
            k = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy=strategy,
                                 subsample=None)
            otr = k.fit_transform(ctr).astype(int).ravel()
            ote = np.clip(k.transform(cte).astype(int).ravel(), 0, nb - 1)
            ne = nb
        except ValueError:
            otr = np.zeros(len(ctr), int); ote = np.zeros(len(cte), int); ne = 1
        nbit = max(ne - 1, 1)
        for b in range(nbit):
            btr.append((otr >= b + 1).astype(np.uint8))
            bte.append((ote >= b + 1).astype(np.uint8))
    return np.column_stack(btr), np.column_stack(bte)


def patterns(X, y):
    A = np.ascontiguousarray(np.column_stack([X, y]).astype(np.uint32))
    return len(np.unique(A.view([('', A.dtype)] * A.shape[1]).ravel()))


def ceiling(X, y):
    """Accuracy of the accuracy-optimal function of X, in-sample on (X,y)."""
    A = np.ascontiguousarray(X.astype(np.uint32))
    V = A.view([('', A.dtype)] * A.shape[1]).ravel()
    u, inv = np.unique(V, return_inverse=True)
    tab = np.zeros((len(u), len(FAMILIES)), np.int64)
    np.add.at(tab, (inv, y), 1)
    pred = tab.argmax(1)[inv]
    return float((pred == y).mean()), float(f1_score(y, pred, average="macro"))


def scan():
    (Xr_tr, ytr), (Xr_te, yte) = raw_capped()
    core, cpss, selm = train_split(ytr)
    rows = []
    for strat in STRATS:
        for nb in BINS:
            t0 = time.time()
            Btr, Bte = booleanize(Xr_tr, Xr_te, nb, strat)
            ptr, pte = patterns(Btr, ytr), patterns(Bte, yte)
            cacc, cf1 = ceiling(Bte, yte)
            Xc, yc = Btr[core].astype(np.float32), ytr[core]
            Xt = Bte.astype(np.float32)
            dt = DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                        random_state=0).fit(Xc, yc)
            rf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=48,
                                        class_weight="balanced_subsample",
                                        n_jobs=32, random_state=0).fit(Xc, yc)
            r = {"strategy": strat, "n_bins": nb, "bits": int(Btr.shape[1]),
                 "patterns_train": ptr, "patterns_test": pte,
                 "ceiling_acc": cacc, "ceiling_f1": cf1,
                 "dt_f1": float(f1_score(yte, dt.predict(Xt), average="macro")),
                 "rf_f1": float(f1_score(yte, rf.predict(Xt), average="macro"))}
            rows.append(r)
            print(f"[scan] {strat:8s} bins={nb:<3} bits={r['bits']:<4} "
                  f"pat(tr/te)={ptr}/{pte:<6} ceiling_acc={cacc:.4f} "
                  f"dt={r['dt_f1']:.4f} rf={r['rf_f1']:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            json.dump(rows, open(f"{OUT}/booleanizer_scan.json", "w"), indent=2)


def run_tm(configs):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    (Xr_tr, ytr), (Xr_te, yte) = raw_capped()
    core, _, _ = train_split(ytr)
    out = []
    for cfg in configs:
        nb, strat = cfg.split(":")
        Btr, Bte = booleanize(Xr_tr, Xr_te, int(nb), strat)
        Btr = Btr.astype(np.uint32); Bte = Bte.astype(np.uint32)  # TMU requires uint32
        per = []
        for seed in SEEDS:
            tm = TMClassifier(number_of_clauses=400, T=320, s=8.0,
                              weighted_clauses=True, platform="CPU", seed=seed)
            t0 = time.time()
            for _ in range(25):
                tm.fit(Btr[core], ytr[core].astype(np.uint32))
            f1 = float(f1_score(yte, tm.predict(Bte), average="macro"))
            per.append(f1)
            print(f"[tm] {cfg} seed{seed} bits={Btr.shape[1]} "
                  f"TEST macro-F1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        out.append({"config": cfg, "bits": int(Btr.shape[1]),
                    "tm_f1_mean": float(np.mean(per)), "tm_f1_sd": float(np.std(per)),
                    "per_seed": per})
        json.dump(out, open(f"{OUT}/booleanizer_tm.json", "w"), indent=2)
        print(f"[done] {cfg} TM={np.mean(per):.4f}+-{np.std(per):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="scan", choices=["scan", "tm"])
    ap.add_argument("--configs", default="10:quantile,30:quantile")
    a = ap.parse_args()
    if a.stage == "scan":
        scan()
    else:
        run_tm([c.strip() for c in a.configs.split(",")])
