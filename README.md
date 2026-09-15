# CAS / STABILIS Empirical Run -- IoT/IoMT dataset comparison

Empirical implementation of **Method 1 (STABILIS)** from
`forensic_fingerprint_methods.md` -- stability-selected clause-activation
signatures for attack-family attribution, on real data, scoped for a single
session, run against **three** different IoT/IoMT datasets to check whether
any one is actually "best" rather than assuming it.

**Primary dataset going forward: IoTM (CICIoMT2024)** (user decision,
2026-08-22). MedSec/WUSTL artifacts are kept on disk for the record but are
no longer the active line of work. CAS here means **per-class** signatures
(one-vs-rest per family) -- the binary normal-vs-attack arm is a fallback,
not the definition of the method. Literature-grounded selection rationale:
`results/DATASET_SELECTION_JUSTIFICATION.md`.

- **`results/RESULTS.md`** -- full primary run on IoTM (CICIoMT2024,
  family-level): what ran vs. documented limitations relative to the full spec.
- **`results/DATASET_COMPARISON.md`** -- the three-way comparison (IoTM vs
  MedSec-25 vs WUSTL-EHMS-2020): which dataset wins on which axis, and why
  STABILIS beats the TM-argmax baseline on one dataset but loses on the
  other two.
- **`results/CAS_PER_CLASS_VS_BINARY.md`** -- per-class CAS feasibility
  answer + the binary (normal-vs-attack) fallback: dedicated 2-class TMs,
  CPSS attack-class signatures with E[V] certificates, 3-seed eval on all
  three datasets, decoded attack evidence lists.
- **`paper/cas_iomt_paper.md`** -- the paper: "Clause-Activation Signatures
  (CAS) for Attack-Family Attribution in IoMT" (IoTM/CICIoMT2024 only;
  `paper/paper.md` is the earlier MedSec-25 study, kept for the record).

**Note on paths:** every script uses absolute paths rooted at
`/FPTM/CAS_IoMT_Empirical` (this repo) and `/FPTM/Datasets/IoTM` (the
dataset). Clone this repo to exactly `/FPTM/CAS_IoMT_Empirical` (or symlink
it there) for the commands below to work unmodified.

## Layout
- `scripts/` -- pipeline code (data prep -> booleanize -> train TM -> CPSS
  signatures -> Task A/B eval -> stability -> figures), per-dataset variants
  suffixed `_medsec` / `_wustl`
- `results/`, `results_medsec/`, `results_wustl/` -- JSON/NPZ metrics, trained
  models, signatures, one dir per dataset
- `figs/` -- PNG figures, including `cross_dataset_comparison.png`

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
python3 scripts/binary_cas.py iotm --seed 42   # also: medsec, wustl; seeds 42/7/123
python3 scripts/binary_cas_aggregate.py
```
