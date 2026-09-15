"""
Deduplicate at the BOOLEAN level, which is where the leakage actually lives.

`dedup_data_prep.py` removes duplicate rows on the 22 raw features and then
booleanizes. The 10-bin quantile thermometer promptly re-creates the
duplicates: the raw-deduped test set has 599,382 rows but only 11,393 distinct
(features,label) codes, so 99.1% of any random split's report half has an exact
twin in its fit half. Deduplicating raw rows cannot fix an encoding collapse.

This collapses the booleanized data itself:
  1. keep one row per distinct (features,label) code within train
  2. same within test
  3. drop test codes whose (features,label) tuple also occurs in train

Patterns carrying more than one label survive as separate rows -- that
ambiguity is real and must not be papered over.

Writes results/bool_dedup_nbins10.npz + a stats json.
"""
import json
import numpy as np

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SRC = "dedup_bool_nbins10.npz"
OUT = "bool_dedup_nbins10.npz"
FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def keys(X, y=None):
    A = X if y is None else np.column_stack([X, y])
    A = np.ascontiguousarray(A.astype(np.uint32))
    return A.view([('', A.dtype)] * A.shape[1]).ravel()


def main():
    d = np.load(f"{BASE}/{SRC}")
    Xtr, ytr = d["Xtr"], d["ytr"].astype(int)
    Xte, yte = d["Xte"], d["yte"].astype(int)

    ktr = keys(Xtr, ytr)
    _, itr = np.unique(ktr, return_index=True)
    itr = np.sort(itr)

    kte = keys(Xte, yte)
    _, ite = np.unique(kte, return_index=True)
    ite = np.sort(ite)

    # drop test codes that also occur in train (label included)
    seen = np.unique(ktr)
    leak = np.isin(kte[ite], seen)
    ite_clean = ite[~leak]

    Xtr2, ytr2 = Xtr[itr], ytr[itr]
    Xte2, yte2 = Xte[ite_clean], yte[ite_clean]

    # how much feature-level overlap remains (patterns seen in train under ANY label)
    ftr = np.unique(keys(Xtr2))
    fte = keys(Xte2)
    feat_overlap = float(np.isin(fte, ftr).mean())

    stats = {
        "source": SRC,
        "train_rows_in": int(len(ytr)), "train_rows_out": int(len(ytr2)),
        "test_rows_in": int(len(yte)), "test_rows_out": int(len(yte2)),
        "test_codes_before_leak_drop": int(len(ite)),
        "test_codes_dropped_as_train_leak": int(leak.sum()),
        "test_feature_pattern_overlap_with_train": round(feat_overlap, 4),
        "train_class_counts": {FAMILIES[i]: int((ytr2 == i).sum()) for i in range(6)},
        "test_class_counts": {FAMILIES[i]: int((yte2 == i).sum()) for i in range(6)},
    }
    print(json.dumps(stats, indent=2), flush=True)

    np.savez_compressed(f"{BASE}/{OUT}", Xtr=Xtr2.astype(np.uint32), ytr=ytr2.astype(np.uint32),
                        Xte=Xte2.astype(np.uint32), yte=yte2.astype(np.uint32))
    json.dump(stats, open(f"{BASE}/bool_dedup_stats.json", "w"), indent=2)
    print(f"[out] {BASE}/{OUT}", flush=True)


if __name__ == "__main__":
    main()
