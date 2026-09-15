"""
Load CICIoMT2024 (family-level, WiFi/MQTT subset) from /FPTM/Datasets/IoTM/,
apply a stratified cap for tractability, and booleanize with per-feature
quantile thermometer encoding (matches forensic_fingerprint_methods.md Sec 0.1).
TMU's feature_negation=True handles literal negation internally, so we only
emit the positive thermometer bits here.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import json
import os

RNG_SEED = 42
TRAIN_CSV = "/FPTM/Datasets/IoTM/Train.csv"
TEST_CSV = "/FPTM/Datasets/IoTM/Test.csv"
OUT_DIR = "/FPTM/CAS_IoMT_Empirical/results"
FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Rate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "DHCP", "ARP", "ICMP", "IGMP",
]
LABEL_COL = "class_name"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
N_BINS = int(os.environ.get("IOTM_N_BINS", 10))
USE_SMOTE = os.environ.get("IOTM_USE_SMOTE", "0") == "1"
# Default OFF: verified regression, not an improvement. SMOTE tested at
# N_BINS=10/clauses=400/T=320/s=8 gave macro-F1=0.6801 vs. 0.6865 without it,
# driven by Benign/Spoofing (already the most confused pair) getting worse:
# Spoofing recall rose to 0.87 at precision 0.27 (was 0.69/0.41), while Benign
# recall fell further to 0.21 (was 0.30). SMOTE's real-valued interpolation
# between minority-class flow features evidently synthesizes points that
# encroach further into the already-overlapping Benign region rather than
# resolving the ambiguity. Kept available via IOTM_USE_SMOTE=1 for the record.

TRAIN_CAP = 40000
TEST_CAP = 15000


def stratified_cap(df, label_col, cap, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for fam in FAMILIES:
        sub = df[df[label_col] == fam]
        n = min(cap, len(sub))
        idx = rng.choice(sub.index.values, size=n, replace=False)
        parts.append(sub.loc[idx])
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_and_sample():
    cols = FEATURE_COLS + [LABEL_COL]
    train_df = pd.read_csv(TRAIN_CSV, usecols=cols)
    test_df = pd.read_csv(TEST_CSV, usecols=cols)

    train_s = stratified_cap(train_df, LABEL_COL, TRAIN_CAP, RNG_SEED)
    test_s = stratified_cap(test_df, LABEL_COL, TEST_CAP, RNG_SEED + 1)

    print("train sample counts:\n", train_s[LABEL_COL].value_counts())
    print("test sample counts:\n", test_s[LABEL_COL].value_counts())
    return train_s, test_s


def clean_inf(train_s, test_s):
    # Rate (and any other feature) can contain +inf from division-by-near-zero
    # flow duration upstream; clip to the finite train max before binning so
    # quantile edges don't collapse to a single degenerate bin.
    for col in FEATURE_COLS:
        finite_max = train_s.loc[np.isfinite(train_s[col]), col].max()
        train_s[col] = train_s[col].replace([np.inf, -np.inf], finite_max)
        test_s[col] = test_s[col].replace([np.inf, -np.inf], finite_max)
    return train_s, test_s


def apply_smote(train_s):
    """Train-only SMOTE (per forensic_fingerprint_methods.md Sec 0.1) to
    balance the minority families (Benign, Spoofing) up to the majority cap,
    synthesizing on the raw continuous features before binarization -- SMOTE
    interpolates between real neighbors in continuous space, which is only
    meaningful pre-binarization, not on the Boolean literals."""
    from imblearn.over_sampling import SMOTE
    X = train_s[FEATURE_COLS].values.astype(np.float64)
    y = train_s[LABEL_COL].map(FAMILIES.index).values
    target = {i: TRAIN_CAP for i in range(len(FAMILIES))}
    # SMOTE requires target >= current count per class; never downsample here
    counts = {i: int((y == i).sum()) for i in range(len(FAMILIES))}
    target = {i: max(target[i], counts[i]) for i in target}
    sm = SMOTE(sampling_strategy=target, random_state=RNG_SEED, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    out = pd.DataFrame(X_res, columns=FEATURE_COLS)
    out[LABEL_COL] = [FAMILIES[i] for i in y_res]
    print("post-SMOTE train counts:\n", out[LABEL_COL].value_counts())
    return out.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)


def booleanize(train_s, test_s):
    train_s, test_s = clean_inf(train_s, test_s)
    if USE_SMOTE:
        train_s = apply_smote(train_s)
    Xtr_raw = train_s[FEATURE_COLS].values.astype(np.float64)
    Xte_raw = test_s[FEATURE_COLS].values.astype(np.float64)

    binarizers = []
    Xtr_bits, Xte_bits = [], []
    literal_names = []
    for j, feat in enumerate(FEATURE_COLS):
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
            # degenerate column (e.g. constant / too few unique quantiles)
            ord_tr = np.zeros(col_tr.shape[0], dtype=int)
            ord_te = np.zeros(col_te.shape[0], dtype=int)
            n_edges = 1

        # thermometer: n_edges-1 cumulative indicator bits, bit k = 1{ord >= k+1}
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

    ytr = train_s[LABEL_COL].map(lambda f: FAMILIES.index(f)).values.astype(np.uint32)
    yte = test_s[LABEL_COL].map(lambda f: FAMILIES.index(f)).values.astype(np.uint32)

    print(f"booleanized: Xtr {Xtr.shape}, Xte {Xte.shape}, literals={Xtr.shape[1]}")
    return Xtr, ytr, Xte, yte, literal_names, binarizers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_s, test_s = load_and_sample()
    Xtr, ytr, Xte, yte, literal_names, binarizers = booleanize(train_s, test_s)

    np.savez_compressed(
        os.path.join(OUT_DIR, "booleanized_data.npz"),
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
    )
    with open(os.path.join(OUT_DIR, "literal_names.json"), "w") as f:
        json.dump({"literal_names": literal_names, "families": FAMILIES, "binarizers": binarizers}, f, indent=2)

    print("Saved booleanized data + literal names to", OUT_DIR)


if __name__ == "__main__":
    main()
