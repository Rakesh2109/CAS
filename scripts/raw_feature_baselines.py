"""
Do the baselines lose anything to the thermometer encoding?

Every comparison so far feeds the ML models the same 154-bit booleanized matrix
the TM consumes, which is the strictly comparable choice. But trees pick their
own split points given raw continuous values, so the 10-bin quantile encoding
may be capping them artificially. This reruns RF/XGB/DT on the RAW 22 features
under the identical leak-free protocol (core -> train, selm -> hyperparameters,
test -> reported once), so the only variable is the input representation.

Row order is reconstructed with data_prep's own stratified_cap at the same seed
and verified against the booleanized labels before use.
"""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from data_prep import FEATURE_COLS, LABEL_COL, FAMILIES
from heldout_protocol import load, train_split, OUT, SEEDS
import nontm_baselines as nb


def raw_capped():
    """Reuse data_prep's own sampling and inf-handling so the row order and the
    inf replacement are identical to what produced booleanized_data.npz
    (test is capped with RNG_SEED + 1, and infinities take the TRAIN max)."""
    from data_prep import load_and_sample, clean_inf
    train_s, test_s = load_and_sample()
    train_s, test_s = clean_inf(train_s, test_s)
    out = []
    for d in (train_s, test_s):
        X = d[FEATURE_COLS].to_numpy(np.float64).astype(np.float32)
        y = d[LABEL_COL].map({f: i for i, f in enumerate(FAMILIES)}).to_numpy(int)
        out.append((X, y))
    return out


def main():
    Xb_tr, ytr, Xb_te, yte = load()
    (Xr_tr, yr_tr), (Xr_te, yr_te) = raw_capped()
    assert np.array_equal(yr_tr, ytr), "train row order does not match booleanized data"
    assert np.array_equal(yr_te, yte), "test row order does not match booleanized data"
    print(f"[align] verified: raw {Xr_tr.shape} matches booleanized {Xb_tr.shape}", flush=True)

    core, cpss, selm = train_split(ytr)
    res = {}
    for name in ("dt", "rf", "xgb"):
        fn, grid = nb.GRIDS[name]
        for rep, (Xtr_, Xte_) in (("bool", (Xb_tr, Xb_te)), ("raw", (Xr_tr, Xr_te))):
            Xc, yc = Xtr_[core].astype(np.float32), ytr[core]
            Xs, ys = Xtr_[selm].astype(np.float32), ytr[selm]
            Xt = Xte_.astype(np.float32)
            per = []
            for seed in SEEDS:
                best, bhp, bm = -1.0, None, None
                for hp_ in grid:
                    m = fn(Xc, yc, seed, hp_)
                    f1 = float(f1_score(ys, m.predict(Xs), average="macro"))
                    if f1 > best:
                        best, bhp, bm = f1, hp_, m
                pt = bm.predict(Xt)
                per.append({"seed": seed, "hp": str(bhp),
                            "test_f1": float(f1_score(yte, pt, average="macro")),
                            "test_acc": float((pt == yte).mean())})
            f = [r["test_f1"] for r in per]; a = [r["test_acc"] for r in per]
            res[f"{name}_{rep}"] = {"test_f1_mean": float(np.mean(f)),
                                    "test_f1_sd": float(np.std(f)),
                                    "test_acc_mean": float(np.mean(a)),
                                    "per_seed": per}
            print(f"[done] {name:4s} {rep:4s}  TEST f1={np.mean(f):.4f}+-{np.std(f):.4f} "
                  f"acc={np.mean(a):.4f}", flush=True)
            json.dump(res, open(f"{OUT}/raw_vs_bool.json", "w"), indent=2)


if __name__ == "__main__":
    main()
