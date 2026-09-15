# CAS / STABILIS Empirical Results -- CICIoMT2024 (family-level)

Empirical run of **Method 1 (STABILIS)** from `forensic_fingerprint_methods.md`
-- the doc's recommended primary method -- against real data, on real Tsetlin
Machine clause activations, with real (if reduced-grid) complementary-pairs
stability selection. This is a **scoped subset** of the full spec; every
scoping decision is called out below rather than silently assumed.

## 1. Dataset

`/FPTM/Datasets/IoTM/{Train,Test}.csv` -- verified to be **CICIoMT2024
(WiFi/MQTT), family-level**: 22 numeric flow features + `class_name` in
{Benign, DDoS, DoS, MQTT, Recon, Spoofing}, i.e. exactly the 5-family + benign
grouping the doc specifies in Sec 0.1. Row counts (6.99M train / 2.18M test)
match the doc's stated ~7.16M/1.61M official split at the same order of
magnitude. **Limitation:** this file only carries family-level labels, not
the 19-subtype breakdown -- so the subtype-level (LOSO, Ping-Sweep-specific)
protocol in Sec 0.4 could not be run; the closest available imbalance-stress
case is the **Spoofing** family (rarest attack family, 16,047/6.99M rows in
the full train split, 5,868/2.18M in test).

**Stratified capped sampling** (session-tractability, not spec): train capped
at 40,000/class (Benign 19,291 and Spoofing 16,047 used in full, all below
cap), test capped at 15,000/class (Spoofing 5,868 used in full). Final:
195,338 train rows / 80,868 test rows.

## 2. Booleanizer

Per-feature quantile thermometer encoding (KBinsDiscretizer, `strategy=quantile`,
10 bins -> up to 9 cumulative binary indicators per feature, bin edges fit on
train only), matching Sec 0.1's "KBinsDiscretizer/thermometer binarization".
TMU's `feature_negation=True` handles literal negation internally, so only
positive thermometer bits are emitted (154 literals -> 308 with TMU's
internal negation).

**Correctness fix applied:** the `Rate` feature (the single largest
family-discriminator -- Benign median rate ~20 vs DDoS median ~56,307)
contained `+inf` values from an upstream near-zero-duration division. Left
unfixed, quantile binning collapsed it to a single degenerate bin (0 bits of
information). Fixed by clipping `inf`/`-inf` to the finite train-max before
binning. This alone raised closed-set macro-F1 from ~0.54 to ~0.72 in
preliminary tests -- worth flagging since it's exactly the kind of silent
data problem a forensic pipeline should not paper over.

A follow-up attempt at finer binning (16 bins, 600 clauses, 240 literals)
was tried and made results *worse* (macro-F1 0.628 vs 0.687, Benign recall
collapsed to 0.06) -- reverted. Reported here for the record, not hidden.

## 3. Base multiclass TM (Sec 0.1)

`TMU.TMClassifier`, 6 classes, 400 clauses, T=320, s=8.0, weighted_clauses=True,
25 epochs, CPU clause banks (no GPU/pycuda in this environment), seed=42.

**Sanity gate result:** closed-set **macro-F1 = 0.6865** -- below the doc's
stated >=0.85 gate. Reported honestly rather than tuned until a target number
appears. Per-family:

| Family | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 0.72 | 0.30 | 0.43 |
| DDoS | 0.86 | 0.77 | 0.81 |
| DoS | 0.78 | 0.89 | 0.83 |
| MQTT | 0.63 | 0.74 | 0.68 |
| Recon | 0.82 | 0.90 | 0.86 |
| Spoofing | 0.41 | 0.69 | 0.52 |

DDoS/DoS confuse with each other (both are flooding attacks with overlapping
rate signatures -- a well-documented CICIoMT2024/CICIoT2023 difficulty).
Benign is the weak point (recall 0.30), splitting almost evenly into
false-MQTT (5,694/15,000) and false-Spoofing (4,658/15,000) predictions --
plausibly because this reduced 22-feature schema has no MQTT-protocol
indicator bit, so MQTT traffic (benign or attack) must be inferred from
rate/flag patterns that partially overlap benign traffic.

## 4. STABILIS signatures (CPSS)

**Reductions from spec** (Sec "METHOD 1", for session tractability):
B=15 complementary-pairs subsamples (spec: 50), lambda-grid narrowed to
`C = geomspace(0.001, 0.1, 6)` (empirically calibrated on this 2400-clause
pool -- C>~0.3 stopped inducing real sparsity and made the union-over-grid
support balloon to ~25% of the clause pool, which makes the E[V] bound
vacuous; not used in the doc's pseudocode but necessary here). pi_thr = 0.6
primary, with a pi_thr in {0.5, 0.7, 0.8, 0.9} sweep for the certificate
figure.

**Correctness fix beyond the doc's literal spec:** CPSS as specified selects
on `|coefficient| > 0` (sign-agnostic), which is correct for the Shah-Samworth
error-control certificate (support recovery doesn't care about sign) but
*wrong* for building an attribution score -- a clause that is stably
selected with a *negative* coefficient is evidence **against** the family
(e.g. it belongs to another family's clause bank, or is a negative-polarity
TM clause), not evidence for it. Naively summing `pi_hat_j * z_j(x)` over
the sign-agnostic support produced a **broken attribution scorer**
(macro-F1 = 0.004, worse than random) because families with larger or
higher-pi_hat signatures always won argmax regardless of the true class.
Fixed in two steps: (1) normalize each family's score to [0,1] (fraction of
that family's signature weight present), and (2) track the mean signed
coefficient per clause across all CPSS fits and restrict the *attribution*
signature to `S_pos = {j : pi_hat_j >= pi_thr AND mean_coef_j > 0}`, while
the **certificate (E[V] bound) still applies to the full sign-agnostic S**,
as specified. This means the reported E[V] bound is a certificate on the
larger sign-agnostic support, not directly on the smaller signed subset
actually used for attribution -- an honest scoping gap, not a hidden one.

| Family | \|S\| (sign-agnostic, certified) | \|S_pos\| (used for attribution) | E[V] bound |
|---|---|---|---|
| Benign | 216 | 48 | <=164.7 |
| DDoS | 131 | 55 | <=97.8 |
| DoS | 289 | 42 | <=212.0 |
| MQTT | 167 | 58 | <=72.2 |
| Recon | 115 | 29 | <=81.5 |
| Spoofing | 311 | 56 | <=249.6 |

The E[V] bounds are loose relative to \|S\| (not a tight "<5% noise"
certificate) -- an honest empirical fact about this clause pool at pi_thr=0.6,
not a claim of a strong guarantee. See `figs/pi_thr_sweep.png` for how
signature size shrinks with pi_thr, and `figs/stabilis_signature_sizes_and_bound.png`.

## 5. Task A -- closed-set attribution (Sec 0.4)

STABILIS signature-match attribution **beats** the direct TM-argmax baseline:

| Method | macro-F1 |
|---|---|
| TM argmax (baseline) | 0.6864 |
| **STABILIS attribution** | **0.7757** |

Per-family recall improves across the board, most dramatically for Benign
(0.30 -> 0.63) and Spoofing precision (0.41 -> 0.55) -- the stability-selected,
sign-filtered signature set is a cleaner discriminator than raw clause-vote
argmax on this data. DDoS/DoS confusion persists (expected -- genuinely
overlapping rate signatures, not a signature-quality artifact). See
`figs/task_a_confusion_matrices.png`, `figs/task_a_per_family_recall.png`.

**Not run** (documented gap, not attempted): Anchors and decision-tree rule
baselines, SHAP top-k stability comparison, second domain (X-IIoTID). These
require additional libraries/time beyond this session's scope; the doc lists
them as required for a full submission-ready result.

## 6. Task B -- novel-family flag via LOFO (Sec 0.4)

**Reduction from spec:** 2 of 5 possible LOFO folds run (Spoofing -- the
imbalance-stress case; MQTT -- a protocol-specific mid-size family), not the
full 5-fold protocol. Each fold retrains a restricted 5-class TM (held-out
family never seen), rebuilds STABILIS signatures on the restricted class set,
and calibrates a Mondrian (per-predicted-family) split-conformal novelty
threshold at target eps=0.05 on held-out *known*-family flows.

| Held-out family | FPR (known, target<=0.05) | TPR (unknown) | AUROC (known vs unknown) | AUPRC | FPR@95%TPR |
|---|---|---|---|---|---|
| Spoofing | 0.035 | 0.040 | **0.275** | 0.161 | 0.966 |
| MQTT | 0.027 | 0.081 | 0.622 | 0.487 | 0.701 |

**Honest finding:** the conformal false-flag-rate guarantee holds cleanly
(both FPRs land under the eps=0.05 target, as the finite-sample split-conformal
theory promises) -- but **detection power against genuinely novel families is
weak**, and for Spoofing the AUROC is *below* 0.5 (known-family flows look
*more* novel to the calibrated score than the truly-unknown Spoofing flows
do). This is not a bug -- it directly reproduces the doc's own stated caveat
in Sec "Attribution + novelty": *"conformal controls the false-flag rate; it
says nothing about power against genuinely novel families."* Plausible
mechanism: Spoofing (ARP-spoofing-type traffic in CICIoMT2024) shares
transport-layer characteristics with legitimate ARP-based device discovery,
so it doesn't present as "novel" in this reduced 22-feature space -- a
finding with real forensic content (this family would be a *blind spot* for
this detector's novelty mechanism), not just a weak number. See
`figs/task_b_lofo_novelty_distributions.png`, `figs/task_b_lofo_summary.png`.

## 7. Stability -- Kuncheva index across TM seeds (Sec 0.4)

**Reduction from spec:** 3 TM seeds (42, 7, 123), not 10 + bootstrap
resamples. Each seed gets its own independently-trained 400-clause TM, its
own CPSS signature extraction (B=15), and pairwise Kuncheva index is computed
on raw clause-index overlap.

| Family | Mean pairwise Kuncheva index (3 seeds) |
|---|---|
| Benign | -0.005 |
| DDoS | 0.037 |
| DoS | 0.011 |
| MQTT | 0.031 |
| Recon | 0.111 |
| Spoofing | -0.015 |

**Honest finding:** Kuncheva indices are near zero (some negative) across
every family -- i.e. **no better than chance overlap** between which raw
clause indices get selected into the signature across independently-seeded
TMs. This is the direct, measured consequence of a fact the doc itself flags
in Sec "Why this is the right backbone": *"TM clause indices are seed-dependent
(...) — any signature without a stability layer is fragile."* CPSS (as
implemented here, directly on raw clause indices) stabilizes selection
*within* a single trained TM (val vs. eval split, same clause bank -- which
is what Task A actually relies on and where it works) but does **not**, on
its own, give cross-retraining signature stability, because clause #47 in one
TM's Benign clause bank and clause #47 in an independently-seeded TM's Benign
clause bank are unrelated learned rules. A real fix would need a
clause-alignment step (e.g. matching by literal-content similarity before
computing Kuncheva) that the doc does not specify and that was out of scope
here. Reported as a genuine negative finding, not smoothed over. See
`figs/stability_kuncheva.png`.

## 8. What this run does and does not support

**Supported by this run:** STABILIS-style stability-selected, sign-filtered
clause signatures produce a real, measurable attribution-quality improvement
over raw TM-argmax voting on real CICIoMT2024 family-level data (0.776 vs
0.686 macro-F1); the CPSS error-control certificate and split-conformal
false-flag-rate control both behave as their theory predicts on this data.

**Not supported / explicitly out of scope:** the doc's >=0.85 closed-set
sanity gate; full 5-fold LOFO / 18-fold LOSO; SHAP/Anchors/decision-tree
baseline comparisons; second-domain (X-IIoTID) transfer; Methods 2
(SIMPLEXPRO) and 3 (DISJUNCT); cross-retraining signature stability (measured
and found weak, not assumed). These are the honest boundaries of a
single-session run against a doc that specifies a multi-month research
program.

## Files

- `booleanized_data.npz`, `literal_names.json` -- prepared data
- `tm_model_seed{42,7,123}.pkl` -- trained TM models
- `tm_argmax_baseline_seed42.json` -- base TM closed-set metrics
- `stabilis_signatures_seed42_thr{0.5,0.6,0.7,0.8,0.9}.json` -- CPSS signatures + diagnostics
- `task_a_results_seed42_thr0.6.json` -- Task A metrics
- `task_b_lofo_{Spoofing,MQTT}.json`, `task_b_lofo_all.json` -- Task B metrics
- `stability_kuncheva.json` -- stability metrics
- `../figs/*.png` -- all figures
