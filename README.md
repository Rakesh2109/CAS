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
```
cd /FPTM/CAS_IoMT_Empirical
python3 scripts/data_prep.py
python3 scripts/train_tm.py --clauses 400 --T 320 --s 8.0 --epochs 25 --seed 42
python3 scripts/stabilis.py 42 0.6 25
python3 scripts/task_a_eval.py 42 0.6
python3 scripts/task_b_lofo.py
python3 scripts/stability_kuncheva.py
python3 scripts/make_figures.py
# binary (normal-vs-attack) CAS fallback arm:
python3 scripts/binary_cas.py iotm --seed 42   # seeds 42/7/123
python3 scripts/binary_cas_aggregate.py
```
