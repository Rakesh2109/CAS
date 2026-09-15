"""
Same grid-search harness as gridsearch_tm.py, but on the paper's curated
22-feature set (a strict subset of the 33 kept features). Purpose: check
whether a wider (clauses, T, s) grid on the *good* feature set can beat the
paper's 0.686 baseline (C=400, T=320, s=8, 25 epochs, KBins-10).

  python3 scripts/gridsearch_tm22.py
"""
import json
import os
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import gridsearch_tm as G

PAPER_22 = [
    "Header_Length", "Protocol Type", "Rate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "DHCP", "ARP", "ICMP", "IGMP",
]
EPOCHS = 25


def prepare22():
    rep = json.load(open(f"{G.BASE}/gridsearch_feature_report.json"))
    kept = rep["kept_features"]
    idx = [kept.index(f) for f in PAPER_22 if f in kept]
    missing = [f for f in PAPER_22 if f not in kept]
    print(f"using {len(idx)}/22 paper features; missing from kept set: {missing}", flush=True)
    d = np.load(f"{G.BASE}/gridsearch_raw.npz")
    Xtr = d["Xtr"][:, idx].astype(np.float64)
    Xte = d["Xte"][:, idx].astype(np.float64)
    ytr, yte = d["ytr"], d["yte"]
    t0 = time.time()
    kb_tr, kb_te = G.thermometer_kbins(Xtr, Xte, G.KBINS)
    tm_tr, tm_te = G.tmu_binarize(Xtr, Xte, G.TMU_BITS)
    print(f"binarized: kbins {kb_tr.shape}, tmu {tm_tr.shape}  ({time.time()-t0:.0f}s)", flush=True)
    G._DATA["kbins"] = (kb_tr, kb_te)
    G._DATA["tmu"] = (tm_tr, tm_te)
    G._DATA["ytr"], G._DATA["yte"] = ytr, yte


def grid():
    g = []
    for binz in ("kbins", "tmu"):
        for c in (400, 700, 1000):
            for f in (0.4, 0.6, 0.8, 1.0, 1.25):
                T = max(1, int(c * f))
                for s in (5.0, 6.0, 8.0, 10.0):
                    g.append((binz, c, T, s, EPOCHS, 42))
    # explicit paper baseline point
    g.append(("kbins", 400, 320, 8.0, EPOCHS, 42))
    return g


def main():
    prepare22()
    grd = grid()
    print(f"22-feat grid: {len(grd)} configs, 64 workers", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(64) as pool:
        results = pool.map(G.train_one, grd)
    results.sort(key=lambda r: -r["best_macro_f1"])
    out = {"n_configs": len(grd), "total_time_s": round(time.time() - t0, 1),
           "feature_set": "paper_22", "results": results}
    with open(f"{G.BASE}/gridsearch_tm22.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== TOP 12 (22-feature) ===")
    print(f"{'bin':6s}{'C':>6}{'T':>6}{'s':>6}{'bestF1':>9}{'ep':>4}")
    for r in results[:12]:
        print(f"{r['binarizer']:6s}{r['clauses']:6d}{r['T']:6d}{r['s']:6.1f}"
              f"{r['best_macro_f1']:9.4f}{r['best_epoch']:4d}")
    print(f"saved {G.BASE}/gridsearch_tm22.json  ({out['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
