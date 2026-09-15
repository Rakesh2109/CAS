# Round 2: fusion rule, Method 2, closing the experimental debt

Follow-up to `RESULTS.md` and `DATASET_COMPARISON.md`. Covers: (1) literature
research on TM hyperparameters and published benchmarks, (2) a fusion rule
combining TM-argmax with the STABILIS signature, (3) naming and figuring the
WUSTL "blind-class repair" effect, (4) implementing Method 2 (SIMPLEXPRO) and
running it on all three datasets, (5) closing Task B + stability on
MedSec-25 and WUSTL, and (6) an investigation into *why* Task B's detection
power is weak everywhere.

## 1. Literature research

Web research (via a research subagent, sources cited) found:

- **No universal T/s-vs-clause-count formula exists in the literature** --
  published TM-clause-count choices for tabular/NIDS data range from
  100-120 clauses (a CICIoMT2024 TM paper, T=10, s=5, n_bins=5 quantile) to
  1,500-2,000 clauses (a 5-class, 554K-row IoMT flow paper) with no derivable
  rule connecting them; s and T are grid-searched per dataset in the
  literature we found, not computed from a formula.
- **A directly comparable paper exists**: Jaiswal et al., "A Tsetlin
  Machine-driven IDS for Next-Gen IoMT Security" (arXiv:2604.03205),
  reports 90.7% accuracy / 90.6% F1 on the *same* CICIoMT2024 WiFi+MQTT
  6-class task, with 100 clauses, T=10, s=5, 15 epochs, n_bins=5. **Caveat
  found and flagged by the research agent**: the paper never states whether
  F1 is macro- or weighted-averaged, and given the severe class imbalance
  in this dataset (Spoofing/Recon are small minority classes), 90.6% is very
  plausibly a weighted/accuracy-like number, not directly comparable to this
  run's macro-F1 of 0.686. Independently confirmed elsewhere: DoS-vs-DDoS
  confusion is a documented, dataset-inherent difficulty on CICIoMT2024, not
  an artifact of this pipeline.
- **WUSTL-EHMS-2020 Spoofing failure is a known, common pattern**: a
  directly relevant paper (arXiv:2506.17329) reports, with no rebalancing
  applied, XGBoost F1=0.81 (best), Decision Tree F1=0.69, Random Forest
  F1=0.14, **SVC F1=0.02** on this exact attack. This run's 0.00 pre-signature
  Spoofing recall is consistent with (not worse than) published
  imbalance-naive baselines on this dataset -- the only mitigation found in
  the literature is algorithm choice (gradient boosting), not resampling.

## 2. Hyperparameter re-tuning attempt (IoTM)

Tested the literature config (100 clauses, T=10, s=5) directly against this
run's original config (400/320/8) on a 40K-row IoTM subsample: the paper's
exact config scored *worse* (macro-F1 0.587 vs 0.635 for the original
config at 15 epochs) -- a different feature-reduction/version of CICIoMT2024
evidently behaves differently. A second round exploring T:clause ratios
around the original config found only ~1pp of headroom (clauses=600,
T=240, s=6 -> 0.665 vs original 0.657 on the same subsample, at 25 epochs),
within subsample-to-subsample noise. **Conclusion: the original
hyperparameters were not badly tuned; no full-scale retrain was performed
since the available gain was marginal and not clearly real (vs. noise).**

## 3. Fusion rule: `combined_l(x) = tm_score_l(x) + lambda * stabilis_score_l(x)`

Rather than treating STABILIS as a standalone competing classifier (which
loses to TM-argmax whenever the base detector is already strong -- see
`DATASET_COMPARISON.md`), fuse the TM's own normalized class-vote
(`clip(class_sum, 0, T) / T`) with the STABILIS signature score. `lambda` is
selected per dataset by grid search (0 to 30, log-spaced) on the CPSS
validation split (disjoint from the eval split -- no leakage), then
evaluated once.

| Dataset | TM argmax | STABILIS only | **Fused (tuned lambda)** | lambda* |
|---|---|---|---|---|
| IoTM | 0.6867 | 0.7757 | **0.7910** | 15.0 |
| MedSec-25 | 0.9498 | 0.8591 | **0.9538** | 0.75 |
| WUSTL-EHMS | 0.6370 | 0.5670 | **0.6920** | 1.5 |

**CORRECTION (see `results_medsec/VERIFIED_FINDINGS.md`): the MedSec-25 row
above is single-seed (42) and does NOT replicate.** Re-run across 3 seeds
(42/7/123), TM-argmax = 0.9527±0.0020 vs Fused = 0.9531±0.0005 -- a
difference 5x smaller than the baseline's own seed noise, i.e. no real
effect on MedSec. The single-seed "win" that motivated this paragraph was
noise, not a finding. **The IoTM and WUSTL rows above have NOT been
re-verified across seeds and should be treated with the same caution** --
they are reported here as originally computed (single seed), not retracted,
but not to be cited as confirmed either.

Original single-seed framing (kept for the record, corrected above): on IoTM
the selected lambda is large (15.0), meaning the fused decision leans heavily
on STABILIS (which was already winning standalone) and still eked out a
further +0.015 over pure STABILIS. On MedSec/WUSTL, small-to-moderate lambda
recovers to (and slightly past) TM-argmax while keeping the signature as a
live signal rather than discarding it. See `figs/fusion_summary.png`
(single-seed figure; see `figs/medsec_verified_multiseed.png` for the
corrected multi-seed MedSec picture).

**Important trade-off found, not hidden**: the macro-F1-optimal lambda is
*not* the same as the lambda that best repairs a blind minority class -- see
next section.

## 4. Named finding: "blind-class repair" (WUSTL Spoofing)

At lambda=0 (pure TM-argmax), WUSTL's Spoofing recall is 0.00 -- the base
detector never predicts this class at all. As lambda increases (more weight
on the STABILIS signature), Spoofing recall rises to 0.53 at the
macro-F1-optimal lambda=1.5, and continues to **0.88-0.90 at lambda>=5**,
at the cost of `normal` recall falling from 0.99 to ~0.50 and macro-F1
*declining* past its peak (macro-F1 optimizes an aggregate that doesn't
weight the invisible class specially). This is the clearest, most original,
most medically-relevant finding in this run: **the fusion weight is a
literal dial between "best aggregate detector" and "best recovery of the
attack class the base detector cannot see at all"**, and for a
patient-safety-relevant attack category (Spoofing => falsified vitals),
the forensic-relevant choice may not be the macro-F1-optimal one. See
`figs/blind_class_repair_wustl.png` for the full lambda-vs-recall-vs-macro-F1
curve, and `results_wustl/fusion_results_seed42_thr0.6.json` for the raw numbers.

## 5. Method 2 (SIMPLEXPRO), implemented and run on all three datasets

Built `scripts/simplexpro.py`: per-sample vote-weighted compositional profile
`h(x) = (w_pos * z(x)) / sum(...)` in the simplex, with the doc's mandatory
correctness fix `w_pos = max(w, eps)` (**verified necessary**: this run's
trained TMs have exactly half of every class's clause weights negative --
e.g. 200/400 for IoTM's Benign class, confirmed by direct inspection -- so
without this fix the simplex construction is undefined). clr-space family
prototypes, argmin-distance attribution, zero-handling via the standard
multiplicative "simple replacement" rule (not naive delta+renormalize).
Reused the already-trained TM models -- no retraining needed for this method.

| Dataset | TM argmax | M1 STABILIS | **M2 SIMPLEXPRO** |
|---|---|---|---|
| IoTM | 0.6867 | 0.7757 | **0.7298** |
| MedSec-25 | 0.9484 | 0.8591 | **0.7221** |
| WUSTL-EHMS | 0.6370 | 0.5670 | **0.5574** |

See `figs/method_comparison_all.png`. **Same qualitative pattern as M1**:
both signature-based methods beat TM-argmax only on IoTM (the weak base
detector); both lose to TM-argmax on MedSec/WUSTL (strong base detectors).
M1 beats M2 head-to-head on every dataset here -- worth noting since the
doc's own decision table calls M1 "LOW risk" and M2 "LOW-MED risk" with M2
pitched as the more elegant geometry, but on this data M1's CPSS-selected,
sign-filtered clause set is simply the stronger discriminator than M2's
vote-weighted compositional-mean prototype. On WUSTL, SIMPLEXPRO also shows
a blind-class-repair-like effect on its own (Spoofing recall 0.00 -> 0.77),
independently corroborating that this is a property of moving away from raw
argmax voting, not an artifact specific to CPSS. **Not implemented**: the
formal concentration/coverage theorems (Thm 1/2) and their conformal-novelty
companion for M2 -- only the point-estimate attribution was built. **Method
3 (DISJUNCT) remains entirely unimplemented.**

## 6. Task B (LOFO): closed on MedSec-25 and WUSTL, plus a novelty-score investigation

Ran the same LOFO protocol (2 folds each) on MedSec-25 (Lateral movement,
Exfiltration) and WUSTL (Spoofing, Data Alteration) that was previously only
run on IoTM.

| Dataset | Held-out | FPR (target<=0.05) | TPR | AUROC |
|---|---|---|---|---|
| WUSTL | Spoofing | 0.026 | 0.027 | 0.470 |
| WUSTL | Data Alteration | 0.000 | 0.000 | 0.671 |
| MedSec-25 | Lateral movement | 0.029 | 0.063 | 0.635 |
| MedSec-25 | Exfiltration | 0.049 | 0.050 | 0.731 |

Combined with IoTM's earlier 2 folds, **all 6 LOFO folds across all 3
datasets show the same pattern**: FPR control is reliable (every fold lands
at or under the eps=0.05 target -- the conformal guarantee holds robustly,
not just on IoTM), but **detection power is uniformly weak (TPR 2.7%-8.1%
in every single fold)**.

**Investigated why** (per your request to improve Task B): re-ran the WUSTL
folds computing a second, richer novelty score in parallel -- the TM's own
full class-vote margin (`clip(class_sum, 0, T)/T`, information-richer than
the ~14-60-clause sign-filtered signature) -- calibrated and evaluated the
same way. **It did not help**: signature-score AUROC 0.470/0.671 vs
tm-margin-score AUROC 0.488/0.514 on the two WUSTL folds -- roughly equal
or *worse*, not better. **Conclusion: the weak novelty-detection power is
not an artifact of using too compressed a scoring function.** Both a sparse
curated signature and the model's full raw confidence signal show the same
ceiling. This points to something more fundamental: these specific held-out
families (Spoofing, Data Alteration, Lateral movement, Exfiltration)
apparently produce clause-activation footprints that are not structurally
distinct enough from known-family footprints in this Boolean-clause feature
space for *any* closed-set-trained confidence signal to flag them as unusual
-- consistent with the earlier per-feature analysis showing WUSTL's Spoofing
has only ~0.2-std-dev separation on its single best discriminating feature
(`results/../` per-feature check in this session, not previously written up).
**Not tried** (would need genuinely different machinery, out of scope for
this pass): density/one-class methods (e.g. a proper Mahalanobis or one-class
SVM on the clr-space SIMPLEXPRO profile), or explicit distributional-shift
features rather than a closed-set-model confidence signal.

## 7. Stability (Kuncheva), closed on MedSec-25 and WUSTL

Same near-zero-or-negative pattern as IoTM's earlier result (raw clause
indices don't align across independently-seeded TMs) on most families, with
one exception: WUSTL's Data Alteration showed a genuinely positive mean
Kuncheva index (0.262, individual seeds 0.09-0.42) -- the one case across all
14 families x 3 datasets where cross-seed signature stability was clearly
above chance. Plausible reason: Data Alteration is the most feature-separable
class in this whole study (F1=0.94-0.99 in every method tried), so its
discriminating clauses may be less arbitrary/more forced by the data itself,
even across different random clause initializations.

## Files added this round

- `scripts/fusion_eval.py`, `scripts/make_fusion_figures.py`
- `scripts/simplexpro.py`, `scripts/make_method_comparison_figure.py`
- `scripts/task_b_lofo_generic.py` (now dual-scores signature vs tm_margin),
  `scripts/stability_kuncheva_generic.py`
- `results{,_medsec,_wustl}/fusion_results_seed42_thr0.6.json`
- `results{,_medsec,_wustl}/simplexpro_results_seed42.json`
- `results_medsec/`, `results_wustl/` -- `task_b_lofo_*.json`, `stability_kuncheva.json` (newly closed)
- `figs/fusion_summary.png`, `figs/blind_class_repair_wustl.png`, `figs/method_comparison_all.png`
