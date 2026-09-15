# CAS -- Clause-Activation Signatures for Attack-Family Attribution in IoMT

Empirical implementation of Clause-Activation Signatures (CAS): weight-based
pruning plus complementary-pairs stability selection (CPSS) on a trained
weighted Tsetlin Machine (WTM), evaluated on **CICIoMT2024** (WiFi/MQTT
subset) for IoMT attack-family attribution.

**Note on paths:** every script uses absolute paths rooted at
`/FPTM/CAS_IoMT_Empirical` (this repo) and `/FPTM/Datasets/IoTM` (the
dataset). Clone this repo to exactly `/FPTM/CAS_IoMT_Empirical` (or symlink
it there) for the commands below to work unmodified.

## Layout
- `scripts/` -- pipeline code: data prep -> booleanize -> train TM -> CPSS
  signatures -> Task A/B eval -> stability -> figures
- `results/` -- JSON metrics, trained models, and signatures from the pipeline
- `figs/` -- figures produced by the pipeline
- `data/sample/` -- small sampled dataset for smoke-testing (see Dataset below)

## Dataset

The scripts read CICIoMT2024 (WiFi/MQTT subset) from `/FPTM/Datasets/IoTM/Train.csv`
and `/FPTM/Datasets/IoTM/Test.csv` (paths are hardcoded, not env-configurable, so
this exact absolute path must exist -- a symlink into your own dataset location
works fine).

- **Full dataset (required to reproduce the paper's numbers):** download
  CICIoMT2024 from the original source (CIC/University of New Brunswick) and
  place the WiFi/MQTT `Train.csv` / `Test.csv` at `/FPTM/Datasets/IoTM/`. Not
  included in this repo -- the two files are 911MB combined, over GitHub's
  practical size limits, and redistribution terms for the raw data aren't
  something we can clear here.
- **`data/sample/`** -- a small, stratified 300/100-rows-per-class sample of
  Train/Test (same 23 columns, same 6 families) checked into this repo for
  smoke-testing the pipeline (`data_prep.py` -> `train_tm.py` -> ...) without
  the full download. It reproduces the pipeline's mechanics, **not** the
  paper's reported macro-F1 -- those numbers require the full capped split
  (40,000 train / 15,000 test rows per class) described in the paper's
  Experimental Setup section. To smoke-test, point `TRAIN_CSV`/`TEST_CSV` in
  `scripts/data_prep.py` at the sample files instead of the full dataset.

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
