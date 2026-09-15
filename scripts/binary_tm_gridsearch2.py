"""
Second binary-TM grid, high-capacity region (user-specified):
  clauses in {500, 1000, 1500, 2000}
  T       in {500, 750, 1000}          (750 covers the explicit C=1000/T=750/s=3 ask)
  s       in {3, 4, 6, 8, 10}
= 60 configs, all benign-oversampled.

Balancing is not an axis here: grid 1 (128 configs) measured balance as by far
the dominant factor -- selection-quarter macro-F1 0.6547 (none) vs 0.7527
(oversample), with all 15 top configs oversampled -- so the unbalanced arm is
dropped to keep this a single parallel wave.

Selection statistic is the per-config MEAN over epoch checkpoints, not the max.
Grid 1 showed the max over 9 checkpoints x 128 configs picks the largest upward
fluctuation: its winner scored 0.8295 at one epoch on a curve averaging 0.725,
and retraining it reproduced 0.658. The report quarter is never touched here.

Output: results/binary_tm_gridsearch2.json
"""
import itertools, json, os, time
import numpy as np
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
BASE = "/FPTM/CAS_IoMT_Empirical/results"
OUT = f"{BASE}/binary_tm_gridsearch2.json"
CLAUSES = [500, 1000, 1500, 2000]
T_VALS = [500, 750, 1000]
S_VALS = [3.0, 4.0, 6.0, 8.0, 10.0]
EPOCH_EVALS = [1, 2, 3, 5, 7, 10, 15]
MAX_EPOCHS = 15
SEED = 42
NPROC = 60


def splits(n):
    p = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return p[:h], p[h:h + q], p[h + q:]


def load():
    from sklearn.metrics import f1_score  # noqa: F401  (warm import in parent)
    d = np.load(f"{BASE}/booleanized_data.npz")
    Xtr, ytr = d["Xtr"], (d["ytr"] != 0).astype(np.uint32)
    Xte, yte = d["Xte"], (d["yte"] != 0).astype(np.uint32)
    rng = np.random.RandomState(SEED)
    mino = np.where(ytr == 0)[0]
    extra = rng.choice(mino, int((ytr == 1).sum()) - len(mino), replace=True)
    idx = rng.permutation(np.concatenate([np.arange(len(ytr)), extra]))
    Xb, yb = np.ascontiguousarray(Xtr[idx]), np.ascontiguousarray(ytr[idx])
    _, sel, _ = splits(len(yte))
    return Xb, yb, Xte[sel], yte[sel]


DATA = None


def init(data):
    global DATA
    DATA = data


def run(cfg):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    from sklearn.metrics import f1_score
    nc, T, s = cfg
    Xf, yf, Xs, ys = DATA
    tm = TMClassifier(number_of_clauses=nc, T=T, s=s, weighted_clauses=True,
                      platform="CPU", seed=SEED)
    t0 = time.time()
    curve = []
    for ep in range(1, MAX_EPOCHS + 1):
        tm.fit(Xf, yf)
        if ep in EPOCH_EVALS:
            curve.append({"epoch": ep,
                          "sel_macro_f1": float(f1_score(ys, tm.predict(Xs), average="macro"))})
    fs = [c["sel_macro_f1"] for c in curve]
    mean = float(np.mean(fs))
    rec = {"clauses": nc, "T": T, "T_ratio": T / nc, "s": s, "balance": "oversample",
           "secs": round(time.time() - t0, 1), "curve": curve,
           "sel_mean": mean, "sel_sd": float(np.std(fs)), "sel_max": float(np.max(fs)),
           "epoch_at_mean": int(min(curve, key=lambda c: abs(c["sel_macro_f1"] - mean))["epoch"]),
           "epoch_at_max": int(max(curve, key=lambda c: c["sel_macro_f1"])["epoch"])}
    print(f"[grid2] C={nc:4d} T={T:4d} s={s:4.1f} -> sel mean {mean:.4f} "
          f"(sd {rec['sel_sd']:.4f}, max {rec['sel_max']:.4f} @ep{rec['epoch_at_max']}) "
          f"{rec['secs']:.0f}s", flush=True)
    return rec


def main():
    data = load()
    print(f"[data] oversampled train={data[0].shape} (attack prev {data[1].mean():.3f}) "
          f"sel={data[2].shape}", flush=True)
    cfgs = list(itertools.product(CLAUSES, T_VALS, S_VALS))
    print(f"[grid2] {len(cfgs)} configs on {NPROC} workers", flush=True)
    with Pool(processes=NPROC, initializer=init, initargs=(data,)) as p:
        res = p.map(run, cfgs, chunksize=1)
    res.sort(key=lambda r: -r["sel_mean"])
    with open(OUT, "w") as f:
        json.dump({"grid": res, "epoch_evals": EPOCH_EVALS, "seed": SEED,
                   "balance": "oversample"}, f, indent=2)
    print("\n=== top 15 by epoch-mean selection macro-F1 ===", flush=True)
    for r in res[:15]:
        print(f"C={r['clauses']:4d} T={r['T']:4d} T/C={r['T_ratio']:.2f} s={r['s']:4.1f} "
              f"mean={r['sel_mean']:.4f} sd={r['sel_sd']:.4f} max={r['sel_max']:.4f} "
              f"ep*={r['epoch_at_mean']}", flush=True)
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
