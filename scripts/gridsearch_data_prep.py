"""
Rebuild CICIoMT2024 (WiFi/MQTT) from the raw per-attack pcap CSVs with the
FULL 38-feature schema, for the TM grid search.

Steps (all fit on the training sample only):
  1. concat all 51 train / 21 test per-attack CSVs, label each row by the
     6-family grouping from its filename
  2. drop leakage-/artifact-prone columns: there are no IP or port fields
     in this schema; we drop IAT (inter-arrival time -- a capture-timing
     feature). Time_To_Live (TTL, a hop count) is kept.
  3. clip +/-inf (Rate divide-by-near-zero) to the finite train max
  4. drop one feature from every pair with |Pearson r| > 0.95 on the
     training sample (greedy, keeps the earlier-listed feature)
  5. stratified cap: 40k/class train, 15k/class test (small classes kept in
     full), shuffle
Saves results/gridsearch_raw.npz (raw continuous features, pre-binarization)
and results/gridsearch_feature_report.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

TRAIN_DIR = "/FPTM/iotm_dataset/CSV/train"
TEST_DIR = "/FPTM/iotm_dataset/CSV/test"
OUT_DIR = "/FPTM/CAS_IoMT_Empirical/results"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
DROP_TIME = ["IAT"]          # capture-timing feature
CORR_THRESH = 0.95
TRAIN_CAP, TEST_CAP = 40000, 15000
SEED = 42


def family_of(fname):
    b = os.path.basename(fname)
    if b.startswith("Benign"):
        return "Benign"
    if b.startswith("ARP_Spoofing"):
        return "Spoofing"
    if b.startswith("MQTT"):
        return "MQTT"
    if b.startswith("Recon"):
        return "Recon"
    if b.startswith("TCP_IP-DDoS"):
        return "DDoS"
    if b.startswith("TCP_IP-DoS"):
        return "DoS"
    raise ValueError(b)


def load_dir(d):
    frames = []
    for fp in sorted(glob.glob(f"{d}/*.csv")):
        fam = family_of(fp)
        df = pd.read_csv(fp)
        df["__family__"] = fam
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def stratified_cap(df, cap, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for i, fam in enumerate(FAMILIES):
        sub = df[df["__family__"] == fam]
        n = min(cap, len(sub))
        idx = rng.choice(sub.index.values, size=n, replace=False)
        parts.append(sub.loc[idx])
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df = load_dir(TRAIN_DIR)
    test_df = load_dir(TEST_DIR)
    print("raw train", train_df.shape, "test", test_df.shape)
    print(train_df["__family__"].value_counts())

    feat_cols = [c for c in train_df.columns if c != "__family__"]
    # drop timing feature(s)
    dropped_time = [c for c in DROP_TIME if c in feat_cols]
    feat_cols = [c for c in feat_cols if c not in dropped_time]

    train_df = stratified_cap(train_df, TRAIN_CAP, SEED)
    test_df = stratified_cap(test_df, TEST_CAP, SEED + 1)

    Xtr = train_df[feat_cols].to_numpy(np.float64)
    Xte = test_df[feat_cols].to_numpy(np.float64)

    # clip inf to finite train max per column
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        fmax = np.nanmax(col[np.isfinite(col)])
        fmin = np.nanmin(col[np.isfinite(col)])
        for X in (Xtr, Xte):
            X[:, j][np.isposinf(X[:, j])] = fmax
            X[:, j][np.isneginf(X[:, j])] = fmin
            X[:, j][np.isnan(X[:, j])] = fmin

    # correlation-based removal (greedy, train only)
    C = np.corrcoef(Xtr, rowvar=False)
    C = np.nan_to_num(C)
    keep = []
    dropped_corr = []
    for j in range(len(feat_cols)):
        redundant = False
        for k in keep:
            if abs(C[j, k]) > CORR_THRESH:
                dropped_corr.append((feat_cols[j], feat_cols[k], round(float(C[j, k]), 4)))
                redundant = True
                break
        if not redundant:
            keep.append(j)

    kept_cols = [feat_cols[j] for j in keep]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]

    ytr = train_df["__family__"].map(FAMILIES.index).to_numpy(np.uint32)
    yte = test_df["__family__"].map(FAMILIES.index).to_numpy(np.uint32)

    np.savez_compressed(f"{OUT_DIR}/gridsearch_raw.npz",
                        Xtr=Xtr.astype(np.float32), ytr=ytr,
                        Xte=Xte.astype(np.float32), yte=yte)
    report = {
        "n_features_raw": len(train_df.columns) - 1,
        "dropped_time": dropped_time,
        "dropped_correlated": dropped_corr,
        "corr_threshold": CORR_THRESH,
        "kept_features": kept_cols,
        "n_kept": len(kept_cols),
        "train_shape": list(Xtr.shape), "test_shape": list(Xte.shape),
        "train_class_counts": {f: int((ytr == i).sum()) for i, f in enumerate(FAMILIES)},
        "test_class_counts": {f: int((yte == i).sum()) for i, f in enumerate(FAMILIES)},
        "note": "no IP or port fields exist in the CICIoMT2024 WiFi/MQTT schema",
    }
    with open(f"{OUT_DIR}/gridsearch_feature_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
