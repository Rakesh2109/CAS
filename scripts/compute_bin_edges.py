"""
Reproduce the exact training-set quantile bin edges used by the paper's
pipeline (data_prep.py: same CSVs, same stratified cap seed, same inf-clip,
same KBinsDiscretizer settings) and print the threshold values behind the
thermometer literals that appear in the paper's decoded-signature table.
Read-only: writes nothing.
"""
import sys

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from data_prep import FEATURE_COLS, clean_inf, load_and_sample

train_s, test_s = load_and_sample()
train_s, test_s = clean_inf(train_s, test_s)

need = {
    "Rate": [3, 5, 6, 7],
    "Header_Length": [3, 4],
    "ICMP": [1],
    "syn_flag_number": [1],
    "rst_flag_number": [1, 2],
}

for j, feat in enumerate(FEATURE_COLS):
    if feat not in need:
        continue
    col = train_s[feat].values.astype(np.float64).reshape(-1, 1)
    nuniq = len(np.unique(col))
    nb = min(10, max(2, nuniq))
    kbd = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile",
                           subsample=None)
    kbd.fit(col)
    edges = kbd.bin_edges_[0]
    print(f"{feat}: n_unique={nuniq}, n_bins={nb}")
    print("  edges:", np.array2string(edges, precision=4, separator=", "))
    for k in need[feat]:
        if k < len(edges):
            print(f"  {feat} >= bin{k}  <=>  {feat} >= {edges[k]:.4f}")
