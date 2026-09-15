# Cross-dataset comparison: is IoTM (CICIoMT2024) actually "best"?

Ran the same pipeline (booleanize -> multiclass TM -> CPSS/STABILIS signature
extraction -> Task A attribution vs TM-argmax baseline) against three
different local IoT/IoMT-labeled datasets to check the claim rather than
assume it. **Answer: no single dataset is "best" on every axis** -- the
three disagree in an interpretable way. Full per-dataset detail for IoTM is
in `RESULTS.md`; this file adds the two new runs and the comparison itself.

## The three datasets

| Dataset | Domain | Features | Classes | Train/test rows used |
|---|---|---|---|---|
| **IoTM** (`/FPTM/Datasets/IoTM/`) | CICIoMT2024, WiFi/MQTT, family-level | 22 reduced flow features | 6 (Benign+5 attack families) | 195,338 / 80,868 |
| **MedSec-25** (`/FPTM/Datasets/medsec/`) | IoMT, kill-chain stage labels | 79 full CICFlowMeter features | 5 (Benign + Recon/Initial-access/Lateral-movement/Exfiltration) | 100,610 / 26,151 |
| **WUSTL-EHMS-2020** (`/FPTM/Datasets/WUSTL/`) | Real medical IoT (EHMS), physiological sensors + network | 35 features incl. Temp/SpO2/Pulse/SYS/DIA/Heart_rate/Resp_Rate | 3 (normal/Spoofing/Data Alteration) | 13,055 / 3,263 |

Same booleanizer (quantile thermometer, 10 bins, TMU internal negation), same
TM hyperparameter family (400 clauses / T=0.8xclauses / s=8.0, except WUSTL
scaled down to 200 clauses given its much smaller size), same CPSS settings
(B=15, `C=geomspace(0.001,0.1,6)`, pi_thr=0.6), same sign-filtered attribution
scorer -- the only thing that changes across runs is the dataset.

## Results

| Dataset | TM-argmax macro-F1 | STABILIS macro-F1 | Delta | Meets doc's 0.85 gate? |
|---|---|---|---|---|
| IoTM (family-level) | 0.686 | **0.776** | **+0.089** | No |
| MedSec-25 (kill-chain) | **0.950** | 0.859 | -0.091 | **Yes** (TM-argmax only) |
| WUSTL-EHMS-2020 | 0.637 | 0.567 | -0.070 | No |

See `figs/cross_dataset_comparison.png`.

## What this actually shows

**MedSec-25 wins on raw closed-set detection quality, by a wide margin.**
0.950 macro-F1 vs IoTM's 0.686 -- and it's the only one of the three that
clears the doc's own >=0.85 sanity gate. The reason is not subtle: MedSec-25
carries the *full* 79-feature CICFlowMeter schema, while IoTM is a reduced
22-feature schema that (as found in `RESULTS.md` Sec 3) doesn't even carry an
explicit MQTT-protocol indicator bit. More raw signal in, better detector out.
Per-family, MedSec-25's weakest class (Lateral movement, F1=0.86) still beats
IoTM's *best* attack-family F1 (Recon, F1=0.86) by a hair and clobbers IoTM's
worst (Benign, F1=0.43).

**But STABILIS's value-add over the raw baseline flips sign depending on the
dataset, and that flip is itself informative.** On IoTM, where the base TM is
mediocre and noisy (0.686), the stability-selected, sign-filtered signature
acts as a *denoiser* and beats raw argmax by +0.089. On MedSec-25, where the
base TM is already excellent (0.950), the same signature construction is
strictly *lossy* relative to the full clause-weighted vote (-0.091) --
compressing a near-perfect voting mechanism down to a sparse pi_hat-weighted
match score throws away signal the raw vote was using. On WUSTL, STABILIS is
also worse on macro-F1 (-0.070) but for a different, more interesting reason:
it trades away most of Data Alteration/normal precision to pull Spoofing
recall from 0.00 to 0.91 (`results_wustl/task_a_results_seed42_thr0.6.json`)
-- a real forensic win on the class that matters most (the attack the base
detector was blind to) at a macro-F1 cost. **This means "beats baseline" is
not a fixed property of the STABILIS method -- it depends on how much slack
there is in the base detector, and on whether you're optimizing macro-F1 or
recall on the class that was previously invisible.**

**WUSTL-EHMS is the most *thematically* on-point IoMT dataset** (real patient
physiological sensor data, e.g. an attack that alters `SYS`/`DIA`/`SpO2`
readings is a direct patient-safety forensic scenario, closer to the doc's
IoMT framing than pure network-flow attacks) but is also the smallest and
most imbalanced (899 Spoofing / 11,418 normal train rows), which shows up
directly as the base TM's Spoofing recall being 0.00 before STABILIS and
still only 0.91-recall-at-0.13-precision after.

## Recommendation

If the goal is the strongest closed-set attribution numbers to report:
**MedSec-25**, by a wide margin -- and it is also the dataset whose
Recon/Initial-access/Lateral-movement/Exfiltration labeling scheme most
directly matches the doc's own "kill-chain stages ... forensic evidence
chains" framing (Sec 0.4's X-IIoTID ask), arguably making it a *better* match
for the doc's actual research questions than IoTM, not just a stronger
number.

If the goal is testing the STABILIS *method's* claimed contribution (does the
error-controlled signature layer add something raw argmax doesn't have):
**IoTM** remains the more informative choice precisely because its base
detector is weak enough for the signature layer's denoising effect to show
up positively -- MedSec-25's near-ceiling base detector leaves no room for
STABILIS to add value on the macro-F1 axis (though it still visibly
reshapes the precision/recall trade-off per class, same as WUSTL).

If the goal is medical-IoMT-authentic *content* (physiological safety
attacks, not just network attacks): **WUSTL-EHMS-2020**, despite the weakest
numbers, is the only one of the three where an attack can directly falsify a
patient's vitals -- worth keeping as a small stress-test case even though it
won't win on any aggregate metric.

**Not run on MedSec-25/WUSTL:** Task B (LOFO novelty) and cross-seed
stability (Kuncheva) -- only Task A (attribution vs baseline) was run on the
two new datasets, to keep this comparison pass tractable. If either dataset
is chosen as primary, the full Task B + stability protocol from `RESULTS.md`
should be re-run on it before treating it as the paper's dataset.

## Files

- `results_medsec/` -- MedSec-25 booleanized data, TM model, signatures, Task A results
- `results_wustl/` -- WUSTL-EHMS booleanized data, TM model, signatures, Task A results
- `figs/cross_dataset_comparison.png` -- the comparison bar chart
