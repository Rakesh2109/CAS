"""
Deduplicated + de-leaked CICIoMT2024 (WiFi/MQTT) prep.

  1. drop exact-duplicate rows within Train and within Test
  2. drop Test rows whose full (features + label) tuple also occurs in Train
     (cross-split leakage)
  3. booleanize with the paper recipe (22 features, per-feature quantile
     thermometer, edges fit on dedup train only, Rate +/-inf -> finite max)

Outputs results/dedup_bool_nbins{10,5}.npz + a stats json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

TRAIN_CSV = "/FPTM/Datasets/IoTM/Train.csv"
TEST_CSV = "/FPTM/Datasets/IoTM/Test.csv"
OUT = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Rate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "DHCP", "ARP", "ICMP", "IGMP",
]
LABEL = "class_name"


def dedup():
    t0 = time.time()
    cols = FEATURE_COLS + [LABEL]
    tr = pd.read_csv(TRAIN_CSV, usecols=cols)
    te = pd.read_csv(TEST_CSV, usecols=cols)
    n_tr0, n_te0 = len(tr), len(te)

    tr_u = tr.drop_duplicates(ignore_index=True)
    te_u = te.drop_duplicates(ignore_index=True)

    # anti-join: keep te_u rows NOT present in tr_u (exact row incl label)
    key = cols
    merged = te_u.merge(tr_u[key].drop_duplicates(), on=key, how="left", indicator=True)
    te_clean = te_u[merged["_merge"].values == "left_only"].reset_index(drop=True)

    stats = {
        "train_raw": n_tr0, "train_dedup": len(tr_u),
        "test_raw": n_te0, "test_dedup": len(te_u),
        "test_leaked_removed": len(te_u) - len(te_clean),
        "test_clean": len(te_clean),
        "train_class_counts": tr_u[LABEL].value_counts().to_dict(),
        "test_clean_class_counts": te_clean[LABEL].value_counts().to_dict(),
        "time_s": round(time.time() - t0, 1),
    }
    print(json.dumps(stats, indent=2), flush=True)
    return tr_u, te_clean, stats


def booleanize(tr, te, nb0):
    Xtr = tr[FEATURE_COLS].to_numpy(np.float64)
    Xte = te[FEATURE_COLS].to_numpy(np.float64)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        fmax = np.nanmax(col[np.isfinite(col)])
        fmin = np.nanmin(col[np.isfinite(col)])
        for X in (Xtr, Xte):
            c = X[:, j]
            c[np.isposinf(c)] = fmax
            c[np.isneginf(c)] = fmax
            c[np.isnan(c)] = fmin
    tr_bits, te_bits, names = [], [], []
    for j, feat in enumerate(FEATURE_COLS):
        ctr, cte = Xtr[:, j:j + 1], Xte[:, j:j + 1]
        nb = min(nb0, max(2, len(np.unique(ctr))))
        try:
            kbd = KBinsDiscretizer(n_bins=nb, encode="ordinal",
                                   strategy="quantile", subsample=None)
            otr = kbd.fit_transform(ctr).astype(np.int16).ravel()
            ote = kbd.transform(cte).astype(np.int16).ravel()
            nbit = max(nb - 1, 1)
        except ValueError:
            otr = np.zeros(len(ctr), np.int16); ote = np.zeros(len(cte), np.int16); nbit = 1
        for k in range(nbit):
            tr_bits.append((otr >= k + 1).astype(np.uint8))
            te_bits.append((ote >= k + 1).astype(np.uint8))
            names.append(f"{feat}>=bin{k+1}")
    Xtr_b = np.column_stack(tr_bits)
    Xte_b = np.column_stack(te_bits)
    ytr = tr[LABEL].map(FAMILIES.index).to_numpy(np.uint8)
    yte = te[LABEL].map(FAMILIES.index).to_numpy(np.uint8)
    np.savez(f"{OUT}/dedup_bool_nbins{nb0}.npz", Xtr=Xtr_b, ytr=ytr, Xte=Xte_b, yte=yte)
    print(f"saved dedup_bool_nbins{nb0}.npz  Xtr {Xtr_b.shape} Xte {Xte_b.shape}", flush=True)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, nargs="+", default=[10, 5])
    args = ap.parse_args()
    tr, te, stats = dedup()
    for nb in args.nbins:
        names = booleanize(tr, te, nb)
    stats["n_literals_nbins10"] = None
    with open(f"{OUT}/dedup_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=int)
    print("done", flush=True)


if __name__ == "__main__":
    main()
