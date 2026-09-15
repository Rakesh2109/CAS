"""
Booleanize the FULL CICIoMT2024 (WiFi/MQTT, family-level) official split --
all 6,987,391 train / 2,175,605 test rows -- with the paper's recipe:
22 numeric features, per-feature quantile thermometer (KBinsDiscretizer,
strategy='quantile', edges fit on train only), +/-inf clip on Rate etc.

Produces, under results/:
  full_bool_nbins{10,5}.npz  -> Xtr (uint8), ytr (0..5), Xte, yte
The 5-bin file is for the binary normal-vs-attack arm; the 10-bin file for
the 6-family multiclass arm. Binary labels are derived at train time as
(y != 0).

  python3 scripts/full_data_prep.py --nbins 10
  python3 scripts/full_data_prep.py --nbins 5
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

TRAIN_CSV = "/FPTM/Datasets/IoTM/Train.csv"
TEST_CSV = "/FPTM/Datasets/IoTM/Test.csv"
OUT_DIR = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Rate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "DHCP", "ARP", "ICMP", "IGMP",
]
LABEL_COL = "class_name"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=10)
    args = ap.parse_args()
    nb0 = args.nbins

    t0 = time.time()
    cols = FEATURE_COLS + [LABEL_COL]
    tr = pd.read_csv(TRAIN_CSV, usecols=cols)
    te = pd.read_csv(TEST_CSV, usecols=cols)
    print(f"loaded train {tr.shape} test {te.shape}  ({time.time()-t0:.0f}s)", flush=True)
    print(tr[LABEL_COL].value_counts().to_dict(), flush=True)

    Xtr = tr[FEATURE_COLS].to_numpy(np.float64)
    Xte = te[FEATURE_COLS].to_numpy(np.float64)
    # clip +/- inf to finite train extremes per column
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        fmax = np.nanmax(col[np.isfinite(col)])
        fmin = np.nanmin(col[np.isfinite(col)])
        for X in (Xtr, Xte):
            c = X[:, j]
            c[np.isposinf(c)] = fmax
            c[np.isneginf(c)] = fmax   # match data_prep.py: inf -> finite train max
            c[np.isnan(c)] = fmin

    tr_bits, te_bits, lit_names = [], [], []
    for j, feat in enumerate(FEATURE_COLS):
        ctr, cte = Xtr[:, j:j + 1], Xte[:, j:j + 1]
        nuniq = len(np.unique(ctr))
        nb = min(nb0, max(2, nuniq))
        try:
            kbd = KBinsDiscretizer(n_bins=nb, encode="ordinal",
                                   strategy="quantile", subsample=None)
            otr = kbd.fit_transform(ctr).astype(np.int16).ravel()
            ote = kbd.transform(cte).astype(np.int16).ravel()
            nbit = max(nb - 1, 1)
        except ValueError:
            otr = np.zeros(len(ctr), np.int16)
            ote = np.zeros(len(cte), np.int16)
            nbit = 1
        for k in range(nbit):
            tr_bits.append((otr >= k + 1).astype(np.uint8))
            te_bits.append((ote >= k + 1).astype(np.uint8))
            lit_names.append(f"{feat}>=bin{k+1}")
        print(f"  {feat}: {nbit} bits  ({time.time()-t0:.0f}s)", flush=True)

    Xtr_b = np.column_stack(tr_bits)
    Xte_b = np.column_stack(te_bits)
    ytr = tr[LABEL_COL].map(FAMILIES.index).to_numpy(np.uint8)
    yte = te[LABEL_COL].map(FAMILIES.index).to_numpy(np.uint8)

    out = f"{OUT_DIR}/full_bool_nbins{nb0}.npz"
    np.savez(out, Xtr=Xtr_b, ytr=ytr, Xte=Xte_b, yte=yte)
    with open(f"{OUT_DIR}/full_bool_nbins{nb0}_literals.json", "w") as f:
        json.dump({"literal_names": lit_names, "families": FAMILIES,
                   "n_train": int(len(ytr)), "n_test": int(len(yte)),
                   "n_literals": int(Xtr_b.shape[1])}, f, indent=2)
    print(f"saved {out}  Xtr {Xtr_b.shape} Xte {Xte_b.shape}  "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
