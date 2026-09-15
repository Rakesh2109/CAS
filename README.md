# CAS -- Clause-Activation Signatures for Attack-Family Attribution in IoMT

Empirical implementation of Clause-Activation Signatures (CAS): weight-based
pruning plus complementary-pairs stability selection (CPSS) on a trained
weighted Tsetlin Machine (WTM), evaluated on **CICIoMT2024** (WiFi/MQTT
subset) for IoMT attack-family attribution.

**Note on paths:** every script uses absolute paths rooted at
`/FPTM/CAS_IoMT_Empirical` (this repo) and `/FPTM/Datasets/IoTM` (the
dataset). Clone this repo to exactly `/FPTM/CAS_IoMT_Empirical` (or symlink
it there) for the commands below to work unmodified.

## Dataset

The scripts read CICIoMT2024 (WiFi/MQTT subset) from `/FPTM/Datasets/IoTM/Train.csv`
and `/FPTM/Datasets/IoTM/Test.csv` (paths are hardcoded, not env-configurable, so
this exact absolute path must exist -- a symlink into your own dataset location
works fine).

- **`data/IoTM_capped_Train.csv` / `data/IoTM_capped_Test.csv`** -- the exact
  dataset used in the paper: the raw CICIoMT2024 WiFi/MQTT split, stratified
  and capped per family (40,000 train / 15,000 test rows per class, seed 42),
  producing the same 195,338 train / 80,868 test rows reported in the paper's
  Experimental Setup. To reproduce the paper's numbers, point `TRAIN_CSV` /
  `TEST_CSV` in `scripts/data_prep.py` at these two files -- no need to
  download the full raw dataset.
- **Full raw dataset (optional):** download CICIoMT2024 from the original
  source (CIC/University of New Brunswick) and place the WiFi/MQTT
  `Train.csv` / `Test.csv` at `/FPTM/Datasets/IoTM/`. Not included in this
  repo -- the two raw files are 911MB combined, well beyond what's needed
  once the capped split above is available.
- **`data/sample/`** -- a much smaller, stratified 300/100-rows-per-class
  sample for quick smoke-testing of the pipeline mechanics only; it does not
  reproduce the paper's reported macro-F1.

## Reproduce
Run in this order -- each stage reads artifacts written by the one before it:
```
cd /FPTM/CAS_IoMT_Empirical
python3 scripts/data_prep.py
python3 scripts/heldout_protocol.py --stage train
python3 scripts/heldout_protocol.py --stage cas
python3 scripts/heldout_protocol.py --stage baselines --train-on core+cpss --features raw --models dt,l1,rf,xgb
python3 scripts/bnn_baseline.py --input raw
python3 scripts/split_ablation.py
python3 scripts/split_ablation.py --pool pruned --out split_ablation_pruned.json
python3 scripts/make_fig_signature_frontier.py
python3 scripts/make_fig_sweep_heldout.py
python3 scripts/make_fig_clause_pruning_heldout.py
```
This reproduces every table and figure in the paper except: the "Input
literals, same readout" ablation row and the sign-agnostic-scorer ablation
(no longer reproducible from checked-in code -- see Known Issues below).

`heldout_protocol.py` implements the leak-free protocol: the TM is retrained
on a `core` split of the training set, CPSS is fit on a disjoint `cpss`
split, operating points are chosen on a disjoint `selm` split, and the test
set is touched exactly once for the final report.

## Known Issues

Verified 2026-09-15 by independently re-deriving every number, table, and
figure in the paper from the underlying arrays/JSON (not just re-reading
cached results). Two findings:

- **One ablation number uses a different, leakier protocol than the rest of
  the paper.** The "Input literals, same readout" row (macro-F1
  0.5876+-0.0112) is produced by `nontm_baselines.py --models cpss_raw`,
  which still uses an old three-way split of the *test set itself*
  (fit/sel/rep) rather than `heldout_protocol.py`'s leak-free split. Per
  `heldout_protocol.py`'s own docstring, that old protocol lets 97.3% of the
  report quarter reappear as an exact feature+label twin in the fit half
  (the 10-bin thermometer encoding collapses the 80,868-row test set to only
  3,592 distinct patterns). No leak-free equivalent of this specific ablation
  currently exists in `scripts/`. Separately, the row's reported accuracy
  (0.6271) matches only seed 42's individual run, not the 3-seed mean
  (0.6340) -- unresolved discrepancy.
- **Two numbers in the paper are not reproducible from any script currently
  in this repo:** Table 3's negative-evidence counts (|S-| = 585 full-pool /
  233 pruned-pool) and the sign-agnostic-scorer ablation (macro-F1
  0.025+-0.032). The full-pool |S-| values exist in a leftover result file
  (`results/heldout/full_signed_seed42/`) but the script that produced it is
  not in `scripts/`; the pruned-pool counts and the sign-agnostic ablation
  have no matching file anywhere in `results/`.

Everything else in the paper -- both TM rows, all CAS full/pruned-pool rows,
both baseline tables, all three data-driven figures, and every dataset/
hyperparameter count -- was confirmed to reproduce exactly (4+ decimal
places) from a from-scratch run of the pipeline above.
