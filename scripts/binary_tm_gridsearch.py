"""
Hyperparameter grid search for the binary (Benign vs Attack) TM.

Motivation: the binary arm of the paper reuses the 6-class settings
(400 clauses / T=320 / s=8 / 25 epochs). Under the binary collapse the
training prior becomes 90.1% attack / 9.9% benign, the Benign clause bank
receives negative (Type-II) feedback on ~90% of samples, and the resulting
detector has Benign recall 0.22 (report-quarter macro-F1 0.629).

Grid axes: clauses x T/clauses ratio x s x class balancing.
Selection: macro-F1 on a stratified 20% validation split held out of Xtr
(train-prior) and, separately, on the test selection quarter used by the
three-way protocol (test-prior). Report quarter is never touched here.

Output: results/binary_tm_gridsearch.json
"""
import itertools, json, os, sys, time
import numpy as np
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
BASE = "/FPTM/CAS_IoMT_Empirical/results"
OUT = f"{BASE}/binary_tm_gridsearch.json"
CLAUSES = [100, 200, 400, 800]
T_RATIO = [0.1, 0.25, 0.5, 0.8]
S_VALS = [2.0, 4.0, 8.0, 16.0]
BALANCE = ["none", "oversample"]
EPOCH_EVALS = [1, 2, 3, 5, 7, 10, 15, 20, 25]
MAX_EPOCHS = 25
SEED = 42


def splits(n):
    perm = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return perm[:h], perm[h:h + q], perm[h + q:]


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro"))


def load():
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xtr, ytr = d["Xtr"], (d["ytr"] != 0).astype(np.uint32)
    Xte, yte = d["Xte"], (d["yte"] != 0).astype(np.uint32)
    rng = np.random.RandomState(SEED)
    val = np.concatenate([rng.choice(np.where(ytr == c)[0],
                                     int(0.2 * (ytr == c).sum()), replace=False)
                          for c in (0, 1)])
    mask = np.ones(len(ytr), bool); mask[val] = False
    _, sel, _ = splits(len(yte))
    return Xtr[mask], ytr[mask], Xtr[val], ytr[val], Xte[sel], yte[sel]


def run(cfg):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    nc, tr_ratio, s, bal = cfg
    T = max(1, int(round(tr_ratio * nc)))
    Xf, yf, Xv, yv, Xs, ys = DATA
    if bal == "oversample":
        rng = np.random.RandomState(SEED)
        mino = np.where(yf == 0)[0]
        need = int((yf == 1).sum()) - len(mino)
        extra = rng.choice(mino, need, replace=True)
        idx = rng.permutation(np.concatenate([np.arange(len(yf)), extra]))
        Xf, yf = np.ascontiguousarray(Xf[idx]), np.ascontiguousarray(yf[idx])
    tm = TMClassifier(number_of_clauses=nc, T=T, s=s, weighted_clauses=True,
                      platform="CPU", seed=SEED)
    t0 = time.time()
    curve = []
    for ep in range(1, MAX_EPOCHS + 1):
        tm.fit(Xf, yf)
        if ep in EPOCH_EVALS:
            curve.append({"epoch": ep,
                          "val_macro_f1": macro_f1(yv, tm.predict(Xv)),
                          "sel_macro_f1": macro_f1(ys, tm.predict(Xs))})
    best = max(curve, key=lambda c: c["val_macro_f1"])
    best_sel = max(curve, key=lambda c: c["sel_macro_f1"])
    rec = {"clauses": nc, "T": T, "T_ratio": tr_ratio, "s": s, "balance": bal,
           "secs": round(time.time() - t0, 1), "curve": curve,
           "best_by_val": best, "best_by_sel": best_sel}
    print(f"[grid] C={nc} T={T} s={s} bal={bal} -> val {best['val_macro_f1']:.4f}"
          f" @ep{best['epoch']} | sel {best_sel['sel_macro_f1']:.4f}"
          f" @ep{best_sel['epoch']} ({rec['secs']}s)", flush=True)
    return rec


DATA = None


def init(data):
    global DATA
    DATA = data


def main():
    data = load()
    print(f"[data] fit={data[0].shape} val={data[2].shape} sel={data[4].shape} "
          f"attack prev fit={data[1].mean():.3f} val={data[3].mean():.3f} "
          f"sel={data[5].mean():.3f}", flush=True)
    cfgs = list(itertools.product(CLAUSES, T_RATIO, S_VALS, BALANCE))
    print(f"[grid] {len(cfgs)} configs", flush=True)
    with Pool(processes=64, initializer=init, initargs=(data,)) as p:
        res = p.map(run, cfgs, chunksize=1)
    res.sort(key=lambda r: -r["best_by_val"]["val_macro_f1"])
    with open(OUT, "w") as f:
        json.dump({"grid": res, "epoch_evals": EPOCH_EVALS, "seed": SEED}, f, indent=2)
    print("\n=== top 15 by validation macro-F1 ===", flush=True)
    for r in res[:15]:
        print(f"C={r['clauses']:4d} T={r['T']:4d} s={r['s']:5.1f} bal={r['balance']:10s} "
              f"ep={r['best_by_val']['epoch']:2d} val={r['best_by_val']['val_macro_f1']:.4f} "
              f"sel={r['best_by_val']['sel_macro_f1']:.4f}", flush=True)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
