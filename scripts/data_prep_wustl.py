"""
Third-dataset comparison run: WUSTL-EHMS-2020 -- genuine medical-IoT
(Enhanced Healthcare Monitoring System) dataset with real patient
physiological readings (Temp, SpO2, Pulse_Rate, SYS, DIA, Heart_rate,
Resp_Rate, ST) alongside network flow features. Small (16,318 rows total)
and only 2 attack categories (Spoofing, Data Alteration) + normal -- a very
different regime from IoTM/MedSec (attacks that manipulate sensor readings,
not just network flooding/recon), used as a third, deliberately small and
different comparison point.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import json
import os

RNG_SEED = 42
CSV = "/FPTM/Datasets/WUSTL/wustl-ehms-2020_with_attacks_categories.csv"
OUT_DIR = "/FPTM/CAS_IoMT_Empirical/results_wustl"
LABEL_COL = "Attack Category"
DROP_COLS = ["Dir", "Flgs", "SrcAddr", "DstAddr", "Sport", "SrcMac", "DstMac",
             "Packet_num", "Label", LABEL_COL]
FAMILIES = ["normal", "Spoofing", "Data Alteration"]
N_BINS = 10


def load_and_split(test_frac=0.2):
    df = pd.read_csv(CSV)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    print("feature cols:", len(feature_cols), feature_cols)
    print(df[LABEL_COL].value_counts())

    rng = np.random.RandomState(RNG_SEED)
    train_parts, test_parts = [], []
    for fam in FAMILIES:
        sub = df[df[LABEL_COL] == fam]
        idx = rng.permutation(sub.index.values)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_parts.append(sub.loc[idx[:n_test]])
        train_parts.append(sub.loc[idx[n_test:]])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=RNG_SEED + 1).reset_index(drop=True)
    print("train counts:\n", train_df[LABEL_COL].value_counts())
    print("test counts:\n", test_df[LABEL_COL].value_counts())
    return train_df, test_df, feature_cols


def booleanize(train_df, test_df, feature_cols):
    Xtr_raw = train_df[feature_cols].values.astype(np.float64)
    Xte_raw = test_df[feature_cols].values.astype(np.float64)

    Xtr_bits, Xte_bits, literal_names, binarizers = [], [], [], []
    for j, feat in enumerate(feature_cols):
        col_tr = Xtr_raw[:, j:j + 1]
        col_te = Xte_raw[:, j:j + 1]
        nuniq = len(np.unique(col_tr))
        nb = min(N_BINS, max(2, nuniq))
        try:
            kbd = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
            ord_tr = kbd.fit_transform(col_tr).astype(int).ravel()
            ord_te = kbd.transform(col_te).astype(int).ravel()
            n_edges = nb
        except ValueError:
            ord_tr = np.zeros(col_tr.shape[0], dtype=int)
            ord_te = np.zeros(col_te.shape[0], dtype=int)
            n_edges = 1
        n_bits = max(n_edges - 1, 1)
        therm_tr = np.zeros((col_tr.shape[0], n_bits), dtype=np.uint8)
        therm_te = np.zeros((col_te.shape[0], n_bits), dtype=np.uint8)
        for k in range(n_bits):
            therm_tr[:, k] = (ord_tr >= (k + 1)).astype(np.uint8)
            therm_te[:, k] = (ord_te >= (k + 1)).astype(np.uint8)
        Xtr_bits.append(therm_tr)
        Xte_bits.append(therm_te)
        literal_names.extend([f"{feat}>=bin{k+1}" for k in range(n_bits)])
        binarizers.append({"feature": feat, "n_bits": n_bits})

    Xtr = np.concatenate(Xtr_bits, axis=1).astype(np.uint32)
    Xte = np.concatenate(Xte_bits, axis=1).astype(np.uint32)
    ytr = train_df[LABEL_COL].map(lambda f: FAMILIES.index(f)).values.astype(np.uint32)
    yte = test_df[LABEL_COL].map(lambda f: FAMILIES.index(f)).values.astype(np.uint32)
    print(f"booleanized: Xtr {Xtr.shape}, Xte {Xte.shape}")
    return Xtr, ytr, Xte, yte, literal_names, binarizers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, test_df, feature_cols = load_and_split()
    Xtr, ytr, Xte, yte, literal_names, binarizers = booleanize(train_df, test_df, feature_cols)
    np.savez_compressed(os.path.join(OUT_DIR, "booleanized_data.npz"), Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    with open(os.path.join(OUT_DIR, "literal_names.json"), "w") as f:
        json.dump({"literal_names": literal_names, "families": FAMILIES, "binarizers": binarizers}, f, indent=2)
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()
