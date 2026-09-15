"""
Non-TM baselines under the CAS three-way protocol.

Answers the reviewer point that CAS is only compared against TM variants.
Every baseline sees the SAME booleanized inputs the TM saw, trains on the
SAME training split, has hyperparameters chosen on the SAME selection
quarter, and is reported once on the SAME report quarter as
`threeway_cas.py` -- the test permutation is RandomState(0), identical.

    train  = Xtr/ytr from booleanized_data.npz   (what the TM trained on)
    fit    = perm[:n/2]        CPSS half-fits (raw-feature ablation only)
    sel    = perm[n/2:3n/4]    hyperparameter selection
    rep    = perm[3n/4:]       reported once, at the end

Class weights are balanced everywhere, since the headline metric is macro-F1
on an imbalanced 6-family problem and CAS's own CPSS uses balanced weights.

    python3 scripts/nontm_baselines.py --models dt,rf,xgb,l1,ebm,rulefit,cpss_raw
"""
import argparse, json, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
SEEDS = [42, 7, 123]
NCLS = 6
KAPPA_GRID = np.geomspace(0.001, 0.1, 6)
PI_THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_SNAPS = (5, 10, 15, 20, 25)
BMAX = 25
N_JOBS = 48


def splits(n):
    """Bit-identical to threeway_cas.splits."""
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def load(npz):
    d = np.load(f"{BASE}/{npz}")
    Xtr = np.asarray(d["Xtr"], dtype=np.float32)
    ytr = np.asarray(d["ytr"], dtype=np.int64)
    Xte = np.asarray(d["Xte"], dtype=np.float32)
    yte = np.asarray(d["yte"], dtype=np.int64)
    return Xtr, ytr, Xte, yte


def balanced_w(y):
    cnt = np.bincount(y, minlength=NCLS).astype(np.float64)
    w = np.zeros(NCLS)
    nz = cnt > 0
    w[nz] = len(y) / (nz.sum() * cnt[nz])
    return w[y]


def ev(model, X, y):
    p = model.predict(X)
    return float(f1_score(y, p, average="macro")), float((p == y).mean())


# ---------------------------------------------------------------- baselines
def fit_dt(Xtr, ytr, seed, hp):
    m = DecisionTreeClassifier(max_depth=hp, class_weight="balanced",
                               random_state=seed)
    return m.fit(Xtr, ytr)


def fit_rf(Xtr, ytr, seed, hp):
    m = RandomForestClassifier(n_estimators=300, max_depth=hp,
                               class_weight="balanced_subsample",
                               n_jobs=N_JOBS, random_state=seed)
    return m.fit(Xtr, ytr)


def fit_l1(Xtr, ytr, seed, hp):
    # One-vs-rest liblinear, matching the per-class OvR that CPSS itself uses.
    base = LogisticRegression(penalty="l1", solver="liblinear", C=hp,
                              max_iter=200, class_weight="balanced",
                              random_state=seed)
    return OneVsRestClassifier(base, n_jobs=NCLS).fit(Xtr, ytr)


def fit_rf_small(Xtr, ytr, seed, hp):
    """Capacity-constrained forest: leaf-count caps regularise far more than
    depth caps under this train->test shift."""
    m = RandomForestClassifier(n_estimators=50, max_leaf_nodes=hp,
                               class_weight="balanced_subsample",
                               n_jobs=N_JOBS, random_state=seed)
    return m.fit(Xtr, ytr)


def fit_xgb(Xtr, ytr, seed, hp):
    from xgboost import XGBClassifier
    depth, nest = hp
    m = XGBClassifier(max_depth=depth, n_estimators=nest, learning_rate=0.1,
                      objective="multi:softmax", num_class=NCLS,
                      tree_method="hist", n_jobs=N_JOBS, random_state=seed,
                      verbosity=0)
    return m.fit(Xtr, ytr, sample_weight=balanced_w(ytr))


def fit_ebm(Xtr, ytr, seed, hp):
    from interpret.glassbox import ExplainableBoostingClassifier
    m = ExplainableBoostingClassifier(interactions=hp, random_state=seed,
                                      n_jobs=N_JOBS)
    return m.fit(Xtr, ytr, sample_weight=balanced_w(ytr))


class RuleFitOvR:
    """imodels RuleFitClassifier is binary; wrap one-vs-rest, argmax on score."""
    def __init__(self, max_rules, seed, cap):
        self.max_rules, self.seed, self.cap = max_rules, seed, cap

    def fit(self, X, y):
        from imodels import RuleFitClassifier
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(y))
        if self.cap and len(y) > self.cap:
            idx = rng.choice(len(y), self.cap, replace=False)
        Xs, ys = X[idx], y[idx]
        self.ms_ = []
        for c in range(NCLS):
            m = RuleFitClassifier(max_rules=self.max_rules, random_state=self.seed)
            m.fit(Xs, (ys == c).astype(int))
            self.ms_.append(m)
        # imodels stores coef as a plain list; np.asarray before comparing
        self.n_rules_ = sum(int((np.asarray(m.coef) != 0).sum()) for m in self.ms_)
        return self

    def predict(self, X):
        S = np.column_stack([m.predict_proba(X)[:, 1] for m in self.ms_])
        return np.argmax(S, 1)


def fit_rulefit(Xtr, ytr, seed, hp):
    return RuleFitOvR(max_rules=hp, seed=seed, cap=40000).fit(Xtr, ytr)


GRIDS = {
    "dt":      (fit_dt,      [4, 6, 8, 10, 12, 16, None]),
    "rf":      (fit_rf,      [None, 12, 20]),
    "rf_small": (fit_rf_small, [24, 48, 96]),
    "l1":      (fit_l1,      list(KAPPA_GRID) + [1.0]),
    "xgb":     (fit_xgb,     [(4, 200), (6, 200), (6, 400), (8, 400)]),
    "ebm":     (fit_ebm,     [0, 10]),
    "rulefit": (fit_rulefit, [100, 200]),
}


def run_supervised(name, Xtr, ytr, Xte, yte, sel, rep, log):
    fn, grid = GRIDS[name]
    per_seed, chosen = [], []
    for seed in SEEDS:
        best, best_hp, best_m = -1.0, None, None
        for hp in grid:
            t0 = time.time()
            m = fn(Xtr, ytr, seed, hp)
            f1s, _ = ev(m, Xte[sel], yte[sel])
            log(f"  [{name}] seed{seed} hp={hp} sel_f1={f1s:.4f} ({time.time()-t0:.0f}s)")
            if f1s > best:
                best, best_hp, best_m = f1s, hp, m
        f1r, accr = ev(best_m, Xte[rep], yte[rep])
        extra = {}
        if name == "dt":
            extra = {"n_leaves": int(best_m.get_n_leaves()),
                     "depth": int(best_m.get_depth())}
        if name == "rulefit":
            extra = {"n_rules_nonzero": int(best_m.n_rules_)}
        if name == "l1":
            extra = {"n_nonzero_coef": int(sum((np.abs(e.coef_) > 1e-8).sum()
                                             for e in best_m.estimators_))}
        per_seed.append({"seed": seed, "hp": str(best_hp), "sel_f1": best,
                         "rep_f1": f1r, "rep_acc": accr, **extra})
        chosen.append(str(best_hp))
        log(f"  [{name}] seed{seed} CHOSE {best_hp} -> rep_f1={f1r:.4f} acc={accr:.4f} {extra}")
    return summarize(name, per_seed, chosen)


def summarize(name, per_seed, chosen):
    f1 = [r["rep_f1"] for r in per_seed]
    ac = [r["rep_acc"] for r in per_seed]
    return {"model": name, "chosen": chosen,
            "rep_f1_mean": float(np.mean(f1)), "rep_f1_sd": float(np.std(f1)),
            "rep_acc_mean": float(np.mean(ac)), "rep_acc_sd": float(np.std(ac)),
            "per_seed": per_seed}


# ------------------------------------------- CAS ablation on raw booleans
def cpss_snap(X, ybin, seed):
    """Same CPSS as threeway_cas.cpss_snap, on raw boolean features."""
    n, m = X.shape
    rng = np.random.RandomState(seed)
    half = n // 2
    sel_c = np.zeros(m, np.int64)
    csum = np.zeros(m, np.float64)
    ng = len(KAPPA_GRID)
    out = {}
    for b in range(BMAX):
        perm = rng.permutation(n)
        for h in (perm[:half], perm[half:2 * half]):
            Xh, yh = X[h], ybin[h]
            if yh.sum() < 2 or (len(yh) - yh.sum()) < 2:
                continue
            u = np.zeros(m, bool)
            for C in KAPPA_GRID:
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                         max_iter=200, class_weight="balanced",
                                         random_state=0)
                clf.fit(Xh, yh)
                cf = clf.coef_.ravel()
                u |= np.abs(cf) > 1e-8
                csum += cf
            sel_c += u
        if (b + 1) in B_SNAPS:
            out[b + 1] = (sel_c / (2 * (b + 1)), csum / (2 * (b + 1) * ng))
    return out


def cas_score(Xev, yev, pis, cfs, thr):
    """Identical decision rule to threeway_cas.score."""
    S = np.zeros((Xev.shape[0], NCLS))
    sizes = []
    for i in range(NCLS):
        pi, cf = pis[i], cfs[i]
        Sp = np.where((pi >= thr) & (cf > 0))[0]
        Sn = np.where((pi >= thr) & (cf < 0))[0]
        sizes.append(len(Sp))
        den = pi[Sp].sum()
        if den <= 0:
            continue
        S[:, i] = (Xev[:, Sp] @ pi[Sp] - (Xev[:, Sn] @ pi[Sn] if len(Sn) else 0)) / den
    pred = np.argmax(S, 1)
    return (float(f1_score(yev, pred, average="macro")),
            float((pred == yev).mean()), sizes)


def run_cpss_raw(Xte, yte, fit, sel, rep, log):
    """CAS machinery (CPSS + signed sums) applied to the 154 raw booleans
    instead of clause activations. Isolates the readout from the clauses."""
    Xf = Xte[fit]
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        snaps = [cpss_snap(Xf, (yte[fit] == i).astype(int), seed + i)
                 for i in range(NCLS)]
        log(f"  [cpss_raw] seed{seed} CPSS done ({time.time()-t0:.0f}s)")
        best, best_key = -1.0, None
        for Bv in B_SNAPS:
            pis = [snaps[i][Bv][0] for i in range(NCLS)]
            cfs = [snaps[i][Bv][1] for i in range(NCLS)]
            for thr in PI_THRS:
                fs, _, _ = cas_score(Xte[sel], yte[sel], pis, cfs, thr)
                if fs > best:
                    best, best_key = fs, (Bv, thr)
        Bv, thr = best_key
        pis = [snaps[i][Bv][0] for i in range(NCLS)]
        cfs = [snaps[i][Bv][1] for i in range(NCLS)]
        fr, ar, sizes = cas_score(Xte[rep], yte[rep], pis, cfs, thr)
        per_seed.append({"seed": seed, "hp": f"B={Bv},thr={thr}", "sel_f1": best,
                         "rep_f1": fr, "rep_acc": ar,
                         "mean_support": float(np.mean(sizes))})
        log(f"  [cpss_raw] seed{seed} CHOSE B={Bv} thr={thr} -> rep_f1={fr:.4f} acc={ar:.4f}")
    return summarize("cpss_raw", per_seed, [r["hp"] for r in per_seed])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dt,l1,rf,xgb,ebm,rulefit,cpss_raw")
    ap.add_argument("--npz", default="booleanized_data.npz")
    ap.add_argument("--out", default="nontm_baselines.json")
    ap.add_argument("--train-on", default="train", choices=["train", "fit", "both"],
                    help="train: Xtr only (what the TM saw). fit: the test fit-half, "
                         "which is what CAS's CPSS is fitted on. both: their union.")
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)

    Xtr, ytr, Xte, yte = load(a.npz)
    fit, sel, rep = splits(len(yte))
    # CAS fits its CPSS readout on the test fit-half, not on Xtr. Baselines given
    # only Xtr are therefore denied labelled in-distribution data that CAS uses,
    # so --train-on fit/both is the information-matched comparison.
    if a.train_on == "fit":
        Xfit, yfit = Xte[fit], yte[fit]
    elif a.train_on == "both":
        Xfit = np.concatenate([Xtr, Xte[fit]]); yfit = np.concatenate([ytr, yte[fit]])
    else:
        Xfit, yfit = Xtr, ytr
    log(f"[data] {a.npz} train_on={a.train_on} fit_set={Xfit.shape} test={Xte.shape} "
        f"fit={len(fit)} sel={len(sel)} rep={len(rep)}")
    Xtr, ytr = Xfit, yfit

    results = {}
    for name in a.models.split(","):
        name = name.strip()
        t0 = time.time()
        log(f"[run] {name}")
        if name == "cpss_raw":
            results[name] = run_cpss_raw(Xte, yte, fit, sel, rep, log)
        else:
            results[name] = run_supervised(name, Xtr, ytr, Xte, yte, sel, rep, log)
        results[name]["wall_s"] = round(time.time() - t0, 1)
        log(f"[done] {name} rep_f1={results[name]['rep_f1_mean']:.4f}"
            f"+-{results[name]['rep_f1_sd']:.4f} ({results[name]['wall_s']}s)")
        with open(f"{BASE}/{a.out}", "w") as f:
            json.dump(results, f, indent=2)
    log(f"[out] {BASE}/{a.out}")


if __name__ == "__main__":
    main()
