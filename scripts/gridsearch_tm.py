"""
TM grid search on the rebuilt 33-feature CICIoMT2024 (WiFi/MQTT, 6 families).

Two binarizers x (clauses, T, s) grid. Each config is trained in its own
single-threaded process; up to N_WORKERS run concurrently.

  binarizer : {"kbins"  -> KBinsDiscretizer quantile thermometer (10 bins),
               "tmu"    -> tmu StandardBinarizer (max_bits_per_feature=10)}
  clauses   : per-class clause count (<= 1000)
  T         : threshold
  s         : specificity

Metric: macro-F1 on the held-out (capped) test split.

  stage 1 (coarse):  python3 scripts/gridsearch_tm.py --stage 1
  stage 2 (refine):  python3 scripts/gridsearch_tm.py --stage 2 --refine-json results/gridsearch_tm_stage1.json
"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer

BASE = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
N_WORKERS = 64
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 25
KBINS = 10
TMU_BITS = 10

# filled by _prepare()
_DATA = {}


def thermometer_kbins(Xtr, Xte, n_bins):
    tr_bits, te_bits = [], []
    for j in range(Xtr.shape[1]):
        col_tr, col_te = Xtr[:, j:j + 1], Xte[:, j:j + 1]
        nuniq = len(np.unique(col_tr))
        nb = min(n_bins, max(2, nuniq))
        try:
            kbd = KBinsDiscretizer(n_bins=nb, encode="ordinal",
                                   strategy="quantile", subsample=None)
            o_tr = kbd.fit_transform(col_tr).astype(int).ravel()
            o_te = kbd.transform(col_te).astype(int).ravel()
            nbit = max(nb - 1, 1)
        except ValueError:
            o_tr = np.zeros(len(col_tr), int)
            o_te = np.zeros(len(col_te), int)
            nbit = 1
        for k in range(nbit):
            tr_bits.append((o_tr >= k + 1).astype(np.uint32))
            te_bits.append((o_te >= k + 1).astype(np.uint32))
    return np.column_stack(tr_bits), np.column_stack(te_bits)


def tmu_binarize(Xtr, Xte, bits):
    import sys
    sys.path.insert(0, "/tmp/claude-0/-FPTM/4822c625-085b-4d7c-bfe0-51629b5ba47d/scratchpad/tmu_repo")
    from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
    b = StandardBinarizer(max_bits_per_feature=bits)
    Xtr_b = b.fit_transform(Xtr).astype(np.uint32)
    Xte_b = b.transform(Xte).astype(np.uint32)
    return Xtr_b, Xte_b


def _prepare():
    d = np.load(f"{BASE}/gridsearch_raw.npz")
    Xtr, ytr, Xte, yte = d["Xtr"].astype(np.float64), d["ytr"], d["Xte"].astype(np.float64), d["yte"]
    t0 = time.time()
    kb_tr, kb_te = thermometer_kbins(Xtr, Xte, KBINS)
    tm_tr, tm_te = tmu_binarize(Xtr, Xte, TMU_BITS)
    print(f"binarized: kbins {kb_tr.shape}, tmu {tm_tr.shape}  ({time.time()-t0:.0f}s)", flush=True)
    _DATA["kbins"] = (kb_tr, kb_te)
    _DATA["tmu"] = (tm_tr, tm_te)
    _DATA["ytr"], _DATA["yte"] = ytr, yte


def train_one(cfg):
    os.environ["OMP_NUM_THREADS"] = "1"
    import sys
    sys.path.insert(0, "/tmp/claude-0/-FPTM/4822c625-085b-4d7c-bfe0-51629b5ba47d/scratchpad/tmu_repo")
    from tmu.models.classification.vanilla_classifier import TMClassifier

    binz, clauses, T, s, epochs, seed = cfg
    Xtr, Xte = _DATA[binz]
    ytr, yte = _DATA["ytr"], _DATA["yte"]
    t0 = time.time()
    tm = TMClassifier(number_of_clauses=clauses, T=T, s=s,
                      weighted_clauses=True, platform="CPU", seed=seed)
    best_f1, best_ep = -1.0, 0
    curve = []
    for ep in range(1, epochs + 1):
        tm.fit(Xtr, ytr)
        if ep % 5 == 0 or ep == epochs:
            pred = tm.predict(Xte)
            f1 = float(f1_score(yte, pred, average="macro"))
            curve.append([ep, round(f1, 4)])
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
    dt = time.time() - t0
    res = {"binarizer": binz, "clauses": clauses, "T": T, "s": s,
           "epochs": epochs, "seed": seed, "best_macro_f1": round(best_f1, 4),
           "best_epoch": best_ep, "final_macro_f1": curve[-1][1],
           "curve": curve, "n_literals": int(Xtr.shape[1]), "time_s": round(dt, 1)}
    print(f"[done {dt:5.0f}s] {binz:5s} C={clauses:4d} T={T:4d} s={s:<4} "
          f"-> best macroF1={best_f1:.4f} @ep{best_ep}", flush=True)
    return res


def stage1_grid():
    grid = []
    for binz in ("kbins", "tmu"):
        for clauses in (200, 500, 1000):
            for ratio in (8, 4, 2):
                T = max(1, clauses // ratio)
                for s in (3.0, 5.0, 8.0, 12.0):
                    grid.append((binz, clauses, T, s, EPOCHS_STAGE1, 42))
    return grid


def stage2_grid(refine_json):
    with open(refine_json) as f:
        prev = json.load(f)
    top = sorted(prev["results"], key=lambda r: -r["best_macro_f1"])[:3]
    grid = []
    seen = set()
    for r in top:
        c0, s0, binz = r["clauses"], r["s"], r["binarizer"]
        for c in sorted({c0, 1000}):
            # explore both the low-T region stage 1 covered and the
            # paper-style high-T region (T up to ~1.5*clauses) it did not
            T_set = {max(1, int(c * f)) for f in (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5)}
            for T in sorted(T_set):
                for s in sorted({round(x, 1) for x in (s0 - 2, s0, s0 + 2, s0 + 4) if x >= 1.5}):
                    key = (binz, c, T, s)
                    if key in seen:
                        continue
                    seen.add(key)
                    grid.append((binz, c, T, s, EPOCHS_STAGE2, 42))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--refine-json", default=f"{BASE}/gridsearch_tm_stage1.json")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    _prepare()
    grid = stage1_grid() if args.stage == 1 else stage2_grid(args.refine_json)
    print(f"stage {args.stage}: {len(grid)} configs, {args.workers} workers", flush=True)

    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(train_one, grid)
    results.sort(key=lambda r: -r["best_macro_f1"])

    out = {"stage": args.stage, "n_configs": len(grid),
           "total_time_s": round(time.time() - t0, 1),
           "feature_report": json.load(open(f"{BASE}/gridsearch_feature_report.json")),
           "results": results}
    out_path = f"{BASE}/gridsearch_tm_stage{args.stage}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== TOP 10 (stage {args.stage}) ===")
    print(f"{'bin':6s}{'C':>6}{'T':>6}{'s':>6}{'lits':>7}{'bestF1':>9}{'ep':>4}")
    for r in results[:10]:
        print(f"{r['binarizer']:6s}{r['clauses']:6d}{r['T']:6d}{r['s']:6.1f}"
              f"{r['n_literals']:7d}{r['best_macro_f1']:9.4f}{r['best_epoch']:4d}")
    print(f"\nsaved {out_path}  ({out['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
