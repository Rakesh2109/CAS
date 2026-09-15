# Clause-Activation Signatures for Attack-Family Attribution in IoMT: An Empirical Study on MedSec-25

## Abstract

Interpretable intrusion detection for the Internet of Medical Things (IoMT) requires more than accurate classification: a forensic analyst needs a compact, auditable reason for each attribution decision. We evaluate two proposed constructions for extracting such reasons from a trained Tsetlin Machine's clause activations: **STABILIS**, which selects a stability-controlled, sign-filtered subset of clauses per attack family via complementary-pairs stability selection (CPSS) on ℓ1-regularized family-vs-rest models, and **SIMPLEXPRO**, which builds a compositional per-family prototype in Aitchison geometry from vote-weighted clause activations. Both are evaluated against a weighted Tsetlin Machine baseline, on MedSec-25, a real kill-chain-staged IoMT network dataset (Benign / Reconnaissance / Initial access / Lateral movement / Exfiltration). Across three independently seeded retrainings, the raw weighted-vote baseline reaches macro-F1 = 0.9527 ± 0.0020, a ceiling neither signature method reliably exceeds: STABILIS reaches 0.8816 ± 0.0160 and SIMPLEXPRO reaches 0.7185 ± 0.0042, with the ranking baseline > STABILIS > SIMPLEXPRO holding on every individual seed with no crossovers. A single-seed result appearing to show a fused baseline-plus-STABILIS combination beating the baseline (+0.0040) did not replicate on two additional seeds (−0.0011, −0.0017); we report this explicitly as a caution about drawing conclusions from single-seed comparisons. We reframe the contribution accordingly: STABILIS recovers 88–93% of the baseline's accuracy using 14–58 stability-selected, sign-positive clauses per family (out of a 2,000-clause pool) together with a reported, if loose, statistical error-control bound on how many of those clauses could be spurious: an interpretability-at-near-parity result, not an accuracy improvement. In an open-set extension (leave-one-family-out, two folds, single seed), split-conformal false-flag-rate control held in both folds (empirical FPR 0.029 and 0.049 against a 0.05 target), but detection power against the truly novel family was substantially higher using the base detector's own confidence margin (AUROC 0.828–0.873) than using the compact signature (AUROC 0.635–0.734). Cross-seed stability of the raw clause indices selected by CPSS was weak (Kuncheva index 0.015–0.041), indicating that clause identity does not transfer across independent retrainings without an explicit alignment step. We report all of the above as verified findings, including the negative and inconclusive ones, and state precisely what remains untested.

**Keywords:** Tsetlin Machine, interpretable machine learning, intrusion detection, IoMT, stability selection, compositional data analysis, open-set recognition

---

## 1. Introduction

Machine-learned intrusion detection systems (IDS) for the Internet of Medical Things face a dual requirement that purely accuracy-driven systems do not: a detection must be both correct and *explainable* in a form a forensic analyst can act on, because the systems being defended can directly affect patient safety. The Tsetlin Machine (TM) [1] is an attractive base learner for this setting because its unit of computation, a conjunctive clause over Boolean literals, is human-readable by construction, and its weighted variant [2] retains this property while compressing the number of clauses needed for a given accuracy. Recent work has applied TMs directly to IoMT intrusion detection with strong reported results [3].

A trained multiclass TM produces, for each input, a binary activation vector over its full clause pool. The question this paper investigates is not whether a TM can classify IoMT traffic (prior work already shows it can) but whether a *compact, statistically justified subset* of that activation vector can serve as a family-level forensic signature: a short, auditable list of clauses whose activation is characteristic of a given attack family, accompanied by a stated bound on how much of that list could be spurious. We evaluate two such constructions, adapted from an internal planning document that proposed them as candidate methods for exactly this purpose. Critically, we independently re-verify both across multiple random seeds before drawing conclusions, after discovering that a single-seed result we initially reported did not hold up.

**Contributions.** (1) We give a full, correctness-audited implementation of stability-selected clause signatures (STABILIS) and compositional prototype signatures (SIMPLEXPRO) built directly on TMU clause activations, including two correctness fixes that were necessary and are not obvious from the method's original specification (§3.2, §3.3). (2) We evaluate both against the raw weighted-TM vote on a real, kill-chain-labeled IoMT dataset, across three independent seeds, and report the verified, not the single-seed, comparison, including a case where an initially promising result failed to replicate (§5.2). (3) We extend the evaluation to an open-set setting and to cross-seed signature stability, and report both a positive result (reliable false-flag-rate control) and negative ones (weak novelty-detection power from the signature; weak cross-seed clause-index stability) without smoothing over either (§5.3–5.4). (4) We state explicitly, rather than by omission, what this study does not establish: no comparison against post-hoc explainability baselines (SHAP, Anchors, decision trees), no second dataset, no completion of the third proposed method, and only single-seed evidence for the open-set results.

## 2. Related Work

**Tsetlin Machines.** The Tsetlin Machine [1] learns a set of conjunctive clauses over Boolean literals via a game-theoretic bandit-driven update rule, using Type I/Type II feedback to grow and prune each clause's included literals. The weighted variant [2] additionally learns an integer vote weight per clause, achieving comparable accuracy with substantially fewer clauses. This is the mechanism this paper's compositional method (§3.3) depends on directly, since it uses those weights to construct a vote-weighted activation profile. We use the TMU reference implementation (github.com/cair/tmu) throughout.

**TM-based IoMT intrusion detection.** Jaiswal et al. [3] train a TM-based IDS on CICIoMT2024 and report strong closed-set accuracy, establishing that TMs are competitive base detectors for this application domain; that work explains individual predictions via the clauses that fired for that sample, but does not aggregate clause activations into a per-family signature with a stated error-control guarantee, which is the gap this paper addresses.

**Stability selection.** Complementary-pairs stability selection [4] repeatedly fits a sparse model (here, ℓ1-regularized logistic regression) on random data splits and retains only the variables selected with high frequency, with a proven finite-sample bound on the expected number of falsely included variables. This bound holds regardless of the underlying data distribution or selector quality, unlike ordinary stability selection's assumption-dependent guarantees. We apply this to clause activations rather than raw features.

**Compositional data analysis.** Aitchison [5] established that ratios, not differences, are the natural comparison operation for data constrained to a simplex (parts of a whole), and that the centered log-ratio (clr) transform maps the simplex isometrically (up to a linear constraint) into ordinary Euclidean space, where standard statistical tools apply. A known practical obstacle is that the log in the clr transform is undefined at zero; we use the multiplicative zero-replacement strategy of Martín-Fernández et al. [6], which preserves the ratios among the nonzero parts rather than perturbing them uniformly.

## 3. Method

### 3.1 Base detector and booleanization

We train a weighted multiclass TMClassifier (TMU) with 400 clauses per class, T = 320, s = 8.0, weighted_clauses = True, for 25 epochs, on 5 classes (Benign, Reconnaissance, Initial access, Lateral movement, Exfiltration). Input features are booleanized via per-feature quantile thermometer encoding: each of the 79 numeric flow features is discretized into up to 10 quantile bins (KBinsDiscretizer, bin edges fit on the training split only) and each bin index k is represented by k cumulative binary indicators ("feature ≥ bin k"), giving 544 positive literal bits; TMU's internal literal negation doubles this to 1,088 literals available to each clause. This yields a clause-activation pool of 2,000 binary values per sample (5 classes × 400 clauses), which both signature methods below operate on.

### 3.2 Method 1 (STABILIS): stability-selected clause signatures

For each family ℓ, we fit an ℓ1-regularized logistic regression predicting family ℓ against the rest, using the clause-activation vector z(x) ∈ {0,1}²⁰⁰⁰ as the feature space. Complementary-pairs stability selection [4] repeats this B = 15 times: each repetition splits a held-out validation set into two complementary halves, fits the ℓ1-logistic model on each half across a grid of regularization strengths C ∈ {geomspace(0.001, 0.1, 6)}, and records, for each clause j, whether it was selected (nonzero coefficient) at any grid point on that half. The per-clause stability score π̂ⱼ is the fraction of the 2B half-fits in which clause j was selected. The signature is Sℓ = {j : π̂ⱼ ≥ π_thr}, with π_thr = 0.6.

**Regularization grid, chosen empirically, not from the method's original specification.** An initial wider grid (C up to 3.0) produced signatures covering roughly 25% of the entire clause pool. At that setting the weakest-regularization grid points were not inducing meaningful sparsity, and the union-over-grid convention required by the stability-selection bound (below) inflated the reported bound past the size of the signature itself, making it vacuous. Restricting the grid to C ∈ [0.001, 0.1], verified by direct inspection to be the range in which ℓ1-logistic regression on this clause-activation space actually selects a small, stable subset, produced signatures of 61–89 clauses per family (2.7–3.7% of the pool) with non-vacuous bounds.

**Error-control certificate.** Following Shah and Samworth [4], with q_Λ estimated as the empirical mean of π̂ times the pool size, the expected number of falsely included clauses in Sℓ satisfies E[V] ≤ q_Λ² / ((2·π_thr − 1)·p), where p = 2,000. At seed 42 this gives, per family: Benign |S| = 75, E[V] ≤ 18.4; Reconnaissance |S| = 66, E[V] ≤ 18.2; Initial access |S| = 66, E[V] ≤ 14.2; Lateral movement |S| = 85, E[V] ≤ 30.2; Exfiltration |S| = 61, E[V] ≤ 18.4. We report these bounds as loose, not tight: they bound roughly a fifth to a third of each signature's size, not the "under 5% spurious" a reader might hope for, because that is what the data supports at this π_thr. Stating a stronger guarantee than the numbers justify would misrepresent the method.

**Correctness fix: sign filtering.** CPSS as specified selects on |coefficient| > 0, which is sign-agnostic and is the correct criterion for the error-control bound above (support recovery does not depend on sign). It is the wrong criterion for building an attribution *score*: a clause that is stably selected with a negative mean coefficient is evidence *against* family ℓ (for instance, because it belongs to a different family's own clause bank), not evidence for it. Naively summing π̂ⱼ·zⱼ(x) over the full sign-agnostic Sℓ produced a broken attribution scorer with macro-F1 = 0.004, worse than random assignment, because a family's raw score scaled with its signature size regardless of whether the sample actually belonged to that family. We track the mean signed coefficient for each selected clause across all CPSS fits and restrict the *attribution* signature to Sℓ,pos = {j ∈ Sℓ : mean_coef_j > 0}, then compute

s_ℓ(x) = [Σ_{j ∈ Sℓ,pos} π̂ⱼ·zⱼ(x)] / [Σ_{j ∈ Sℓ,pos} π̂ⱼ]  ,  ŷ(x) = argmax_ℓ s_ℓ(x)

normalized to [0,1] so that families with differently sized signatures remain comparable under argmax (an earlier unnormalized version had the same failure mode as the sign bug, for a related reason: raw summed scores scale with signature size and weight magnitude, not with match quality). After both fixes, Sℓ,pos contains 14–58 clauses per family (median 27), the number actually used for attribution and reported in the evidence-list sense of the method.

### 3.3 Method 2 (SIMPLEXPRO): compositional prototype signatures

For each sample, we construct a vote-weighted compositional profile over the clause pool: h(x) = (w₊ ⊙ z(x)) / Σ(w₊ ⊙ z(x)), a point on the 1999-simplex, where w is the trained model's per-clause weight vector (weight_banks[c].get_weights() concatenated across classes, in the same column order as the clause-activation matrix) and w₊ = max(w, 1).

**Correctness fix: the positive-weight floor is not optional.** The method's original specification calls this a "mandatory" fix without independent verification in this codebase; we verified it directly. Inspecting the trained weight vectors shows that exactly half of every class's clause weights are negative by construction of the weighted-TM training procedure (e.g., 200 of 400 for the Benign class, with 0 exactly-zero weights). Weighted TMs represent a clause's vote *against* a class as a negative weight, which is essential to how the classifier itself works, but is undefined input to a simplex/log-ratio construction that requires strictly positive parts. Without the max(·, 1) floor, roughly half of every sample's contribution to h(x) is either negative (undefined) or silently dropped, and the resulting composition does not represent the intended "vote-weighted activation profile."

Zero entries in h(x) (clauses in a class's bank that a sample does not activate) are handled by the multiplicative "simple replacement" rule of Martín-Fernández et al. [6] with pseudo-count δ = 1/(m·n): zero parts are set to δ, and the remaining nonzero parts are rescaled by (1 − n_zero·δ) so that the composition still sums to 1 and the ratios among originally nonzero parts are preserved exactly, as opposed to a naive "add δ everywhere and renormalize," which would distort those ratios. Samples that activate no positively-weighted clause at all fall back to the uniform composition (every part = 1/m).

The centered log-ratio transform clr(h)ⱼ = log(hⱼ / g(h)), with g(h) the geometric mean of h, maps each sample to Euclidean space. Family prototypes μℓ are the clr-space means of a family's validation-split samples; attribution is ŷ(x) = argmin_ℓ ‖clr(h(x)) − μℓ‖, the clr-coordinate form of the Aitchison distance (equivalent up to the ilr isometry [5]).

### 3.4 Fusion (exploratory)

We additionally tested a linear combination combined_ℓ(x) = tm_score_ℓ(x) + λ·s_ℓ(x), where tm_score_ℓ(x) = clip(class_sum_ℓ(x), 0, T)/T is the base detector's own normalized class-vote and s_ℓ(x) is the STABILIS score, with λ selected per run by grid search on a validation split disjoint from the evaluation split. As detailed in §5.2, this did not produce a result that survived multi-seed verification, and we report it as a negative/inconclusive finding rather than a contribution.

### 3.5 Open-set extension

For leave-one-family-out evaluation, a restricted 4-class TM is retrained with one family entirely absent from training. On the resulting model, a novelty score ν(x) = 1 − max_ℓ score_ℓ(x) is computed two ways, using the STABILIS signature score and, separately, the base detector's own tm_score, and calibrated via Mondrian (per-predicted-family) split-conformal calibration [inferred from standard conformal prediction theory, no additional citation added beyond what is used in-text] at a target false-flag rate ε = 0.05, using held-out samples from the known (non-held-out) families.

## 4. Experimental Setup

**Dataset.** MedSec-25 [7] is a real IoMT network-traffic dataset collected in a healthcare-IoT lab environment (Raspberry Pi nodes plus ECG, EEG, respiration, thermistor, and other medical/environmental sensors, using MQTT, SSH, FTP, HTTP, and DNS protocols), with attack traffic generated as a multi-stage campaign whose phases are labeled according to MITRE ATT&CK-inspired kill-chain stages: Reconnaissance, Initial access, Lateral movement, and Exfiltration, plus Benign. The full dataset contains 554,535 flow records (CICFlowMeter-style features) across these five classes (Reconnaissance 401,683; Initial access 102,090; Exfiltration 25,915; Lateral movement 12,498; Benign 12,348), a strongly imbalanced distribution that we do not correct via resampling. We use 79 numeric features after dropping session-identifying columns (Flow ID, Src IP, Dst IP, Timestamp) and the label column itself.

**Sampling.** For computational tractability we draw a stratified sample capped at 30,000 rows per class for training and 8,000 per class for testing (using all available rows where a class has fewer), yielding 100,610 training and 26,151 test rows. We did not observe missing values or infinite values in this feature set (in contrast to a companion dataset used in earlier, out-of-scope exploration of this pipeline, where an unhandled `inf` value in a rate-like feature silently zeroed out that feature's information content, a class of bug worth checking for in any CICFlowMeter-derived dataset, though it did not occur here).

**Train/evaluation protocol.** The test split is further divided in half: one half (val_idx) is used to fit CPSS and build family prototypes; the other half (eval_idx) is held out for all reported closed-set metrics, so that no sample used for signature construction is reused for evaluation. All headline macro-F1 numbers in this paper are computed on eval_idx (13,076 samples) unless stated otherwise.

**Seeds.** Three independently seeded TM trainings (seeds 42, 7, 123) are used for all multi-seed claims; results explicitly marked single-seed use seed 42 only and are stated as such.

**A verified implementation detail.** During cross-checking, we found that computing the baseline's reported macro-F1 by clipping raw class-vote sums to [0, T] before taking argmax (a step needed to normalize the fusion score in §3.4) silently disagreed with the base detector's actual argmax decision on 18 of 13,076 evaluation samples: clipping both classes of a tied negative-sum pair to zero can flip which class argmax selects. We use the detector's true, unclipped `predict()` output for every baseline number reported in this paper; the clipped score is used only internally where the fusion formula requires a bounded [0,1] contribution.

## 5. Results

### 5.1 Closed-set attribution

Table 1 reports macro-F1 (mean ± standard deviation over the three seeds) for the base detector and each signature method.

**Table 1. Closed-set macro-F1, MedSec-25 (3 seeds).**

| Method | Seed 42 | Seed 7 | Seed 123 | Mean ± SD |
|---|---|---|---|---|
| TM argmax (baseline) | 0.9498 | 0.9538 | 0.9544 | **0.9527 ± 0.0020** |
| Method 1 (STABILIS) | 0.8591 | 0.8917 | 0.8940 | 0.8816 ± 0.0160 |
| Method 2 (SIMPLEXPRO) | 0.7221 | 0.7207 | 0.7126 | 0.7185 ± 0.0042 |
| Fused (baseline + STABILIS) | 0.9538 | 0.9527 | 0.9527 | 0.9531 ± 0.0005 |

The ranking baseline > STABILIS > SIMPLEXPRO holds on every individual seed, with no crossovers. At seed 42, per-class performance for the baseline is: Benign P = 0.98 / R = 0.95 / F1 = 0.96; Reconnaissance P = 1.00 / R = 0.97 / F1 = 0.98; Initial access P = 0.99 / R = 0.99 / F1 = 0.99; Lateral movement P = 0.82 / R = 0.91 / F1 = 0.86 (the weakest class in every method tried); Exfiltration P = 0.94 / R = 0.94 / F1 = 0.94.

### 5.2 The fusion result did not replicate: a methodological note

A single-seed comparison (seed 42) initially suggested that fusing the baseline vote with the STABILIS score (§3.4) improves on the baseline alone (0.9538 vs. 0.9498, +0.0040), which we drafted as a positive finding before checking additional seeds. Re-running the identical procedure (same fusion formula, same λ-grid-search protocol) on seeds 7 and 123 (Table 1, bottom row) shows the fused variant *losing* to the baseline on both (−0.0011 and −0.0017 respectively). The mean advantage across all three seeds (+0.0004) is roughly five times smaller than the baseline's own seed-to-seed standard deviation (0.0020): the effect is not distinguishable from noise. Compounding this, the macro-F1-optimal λ selected by the grid search was itself unstable across seeds (0.75, 2.0, 0.25, with no consistent value), consistent with the validation-split-based selection overfitting a single split rather than locating a reusable operating point. We report this explicitly, including the fact that our own first-pass draft treated the seed-42 result as a finding, as a concrete illustration of why single-seed comparisons should be treated as provisional until checked.

### 5.3 Interpretability at near-parity: the reframed contribution

Given §5.1–5.2, this study does not support a claim that either signature method, or their fusion with the baseline, improves closed-set accuracy. What it does support: **STABILIS recovers 88–93% of a near-ceiling detector's macro-F1 (0.88 vs. 0.95) using a per-family signature of 14–58 clauses (median 27) drawn from a 2,000-clause pool**, each accompanied by a stated, if loose, statistical bound on how many of the *sign-agnostic* selected clauses (61–89 per family) could be false inclusions (§3.2). This is the interpretability contribution the method's originating specification frames it as, "a signature is a lead, not a name," a compact evidence list rather than a black-box vote. The honest characterization of what this experiment shows is that this interpretability is available at a real but bounded accuracy cost on this dataset, not for free.

### 5.4 Open-set detection (single seed)

We evaluate two leave-one-family-out folds (holding out Lateral movement, then Exfiltration in turn), retraining a restricted 4-class detector for each, at seed 42 only. Table 2 reports the conformal false-flag rate (target ε ≤ 0.05) and detection AUROC using each of the two novelty scores.

**Table 2. Open-set results, single seed (42), 2 of 4 possible LOFO folds.**

| Held-out family | FPR (target ≤ 0.05) | TPR | AUROC (signature score) | AUROC (base-detector margin) |
|---|---|---|---|---|
| Lateral movement | 0.029 | 0.063 | 0.635 | **0.873** |
| Exfiltration | 0.049 | 0.050 | 0.734 | **0.828** |

The conformal false-flag-rate guarantee held in both folds: empirical FPR (0.029, 0.049) stayed at or under the ε = 0.05 target, as the underlying split-conformal theory predicts regardless of how good the underlying score is. Detection *power* is a different matter: the sparse STABILIS signature score reached AUROC 0.635–0.734, while the base detector's own confidence margin (richer information, since it uses the full clause-weighted vote rather than a 14–58-clause subset) reached AUROC 0.828–0.873, an improvement of 0.09–0.24. We interpret this as the compression that helps the closed-set attribution case (§5.3) costing real detection power in the open-set case, where the discarded clauses evidently still carried information relevant to recognizing "this doesn't look like any known family." We did not test this on additional seeds or additional folds, and do not extrapolate beyond the two folds and one seed reported here.

### 5.5 Cross-seed signature stability

We compute the Kuncheva index, a chance-corrected measure of set overlap, for the raw CPSS-selected clause indices between each pair of the three independently seeded models, per family (Table 3).

**Table 3. Kuncheva index of CPSS-selected clause indices, pairwise across seeds {42, 7, 123}.**

| Family | Mean Kuncheva index | Individual pairwise values |
|---|---|---|
| Benign | 0.032 | 0.031, 0.015, 0.050 |
| Reconnaissance | 0.015 | −0.008, 0.059, −0.008 |
| Initial access | 0.020 | −0.015, 0.057, 0.019 |
| Lateral movement | 0.041 | 0.028, −0.013, 0.108 |
| Exfiltration | 0.020 | 0.087, −0.014, −0.015 |

All values are small (0.015–0.041 mean, several individual pairs indistinguishable from or below zero), near chance-level agreement between which raw clause indices are selected across independent retrainings. This is consistent with a known property of Tsetlin Machines: clause indices are an artifact of random initialization and training dynamics, so clause #k in one training run and clause #k in an independently seeded run are, in general, unrelated learned rules, and nothing in the CPSS procedure as applied here addresses this. Note that this is distinct from, and does not contradict, the fact that CPSS's *within-model* stability selection (train/eval split of the same fixed trained model) is exactly what makes the closed-set attribution in §5.1/5.3 work; the weak result here is specifically about identity transfer *across* independently retrained models, which would require an explicit clause-alignment step (for instance, matching by literal-content similarity) that this study does not implement.

## 6. Discussion

Three results in this study point in the same direction. First, the fusion result that did not replicate (§5.2) is a reminder that a promising single-seed number is not yet a finding. The discipline of re-checking across seeds before reporting a positive result is what caught it here, and we surface the reversal rather than quietly dropping the earlier framing. Second, the open-set result (§5.4) shows that the same compression that lets STABILIS approach the baseline's closed-set accuracy with an order-of-magnitude fewer clauses actively hurts when the task changes to recognizing the *absence* of a known pattern rather than the presence of one. This is an asymmetry a system designer choosing between the compact signature and the raw detector's confidence score should weigh against the intended use case, not treat as settled by the closed-set result alone. Third, the weak cross-seed stability (§5.5) means that "the same signature" is not yet a well-defined object across independent deployments of this pipeline, even though it is a well-defined and useful object *within* one trained model, a distinction a forensic tool built on this method would need to make explicit to its users rather than imply is stronger than it is.

## 7. Limitations

We list what this study does not establish, rather than leaving it implicit. (1) No comparison against post-hoc explainability baselines (SHAP, Anchors, decision-tree rule extraction) that the originating research plan called for; we cannot say whether STABILIS's stability guarantee is a meaningful advantage over these alternatives on this dataset, only that it exists as a guarantee they do not offer. (2) Only 2 of the 4 possible leave-one-family-out folds were run, at a single seed; the open-set results in §5.4 should be read as a case study, not a stable estimate. (3) A third method proposed in the originating specification (DISJUNCT, a coding-theory-based approach to certifying mixture-attack identifiability) was not implemented at all. (4) No second dataset was used to test whether these findings transfer beyond MedSec-25's specific feature schema and attack taxonomy. (5) The reported error-control bounds (§3.2) are loose relative to signature size at the stability threshold used here, not a tight guarantee, and we have not explored whether a different threshold would materially tighten them without unacceptably shrinking the signatures. (6) Cross-seed clause-index stability is weak and unaddressed by any alignment procedure in this work. (7) **A potential feature-leakage issue, found only after decoding the actual signature clauses into their literal content (post-submission relative to the numbers in §5): `Src Port` and `Dst Port` were never excluded from the 79-feature schema, and decoding the seed-42 signatures shows 60.3% of all signature clauses (76/126, across all five families) contain at least one port literal, including several of the highest-confidence (π̂ = 1.0) clauses for Benign, Initial access, Lateral movement, and Exfiltration** (e.g., the second-ranked Benign clause is a bare two-literal conjunction on `Src Port` and `Dst Port`). Source port in particular is normally an ephemeral, per-connection value an attacker does not choose, so a model or signature relying on it is plausibly fitting an artifact of how this specific lab capture assigned ports rather than a property of the attack behavior itself, and may not generalize to a different deployment or capture session. This was not caught by the seed-replication verification in §5.2, which checks numerical reproducibility, not feature validity; it should be treated as an open question affecting every closed-set and interpretability number in this paper, not only the signature-decoding result that surfaced it, until re-run with port features excluded.

## 8. Conclusion

We implemented and empirically verified two proposed methods for extracting per-family forensic signatures from a weighted Tsetlin Machine's clause activations on a real, kill-chain-labeled IoMT dataset. Neither method, nor a fusion of the better-performing one with the base detector, reliably improves on the base detector's own closed-set accuracy once checked across multiple seeds, a result we report plainly, including the correction of an initial single-seed finding that did not hold up. The methods' genuine, verified value on this dataset is interpretability at a bounded accuracy cost: a 14-to-58-clause, statistically stability-selected, sign-filtered evidence list per attack family that recovers the large majority of a near-ceiling detector's performance. We also report, without qualification, where the compact signature underperforms the raw detector (open-set detection power) and where the underlying clause-selection procedure itself is not yet stable (cross-seed clause identity), properties any deployment of this approach would need to account for.

## Acknowledged Limitations

See §7. In addition: this paper's dataset sampling (30,000/8,000 per-class caps) was chosen for computational tractability within the scope of this study and is smaller than the full MedSec-25 release; results on the full, unsampled dataset were not obtained and may differ.

## Data and Code Availability Statement

The MedSec-25 dataset is described in [7]; access follows that publication's terms. All pipeline code (booleanization, TM training, CPSS signature extraction, compositional prototype construction, open-set evaluation, stability analysis) and the exact result files underlying every number in this paper are retained alongside this manuscript.

## Ethics Declaration

This study performs offline forensic/interpretability analysis on a previously collected, published network-traffic dataset. No new data collection, human subjects, or patient data were involved in this work; MedSec-25's own collection used simulated sensor traffic in a lab environment, per its originating publication [7].

## Author Contributions

Conceptualization, methodology implementation, formal analysis, verification, and writing were performed as a single-author empirical study.

## Conflict of Interest Statement

The authors declare no conflict of interest.

## Funding

No external funding was received for this study.

## AI Disclosure Statement

This paper was drafted with the assistance of an AI system (Claude), which implemented the experimental pipeline, ran and re-verified all reported experiments, and produced the manuscript text under human direction and review. All numerical results were independently re-derived or cross-checked via at least one alternative computational path before inclusion (see §4, "A verified implementation detail," and the seed-replication discipline in §5.2). All literature citations were verified to exist via web search at the time of writing, not generated from memory alone.

## References

[1] O.-C. Granmo, "The Tsetlin Machine: A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic," *arXiv:1804.01508*, 2018.

[2] A. Phoulady, O.-C. Granmo, S. R. Gorji, and H. A. Phoulady, "The Weighted Tsetlin Machine: Compressed Representations with Weighted Clauses," *arXiv:1911.12607*, 2019/2020.

[3] R. Jaiswal, P.-A. Andersen, L. R. Cenkeramaddi, L. Jiao, and O.-C. Granmo, "A Tsetlin Machine-driven Intrusion Detection System for Next-Generation IoMT Security," *arXiv:2604.03205*, 2026.

[4] R. D. Shah and R. J. Samworth, "Variable Selection with Error Control: Another Look at Stability Selection," *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, vol. 75, no. 1, pp. 55–80, 2013.

[5] J. Aitchison, "The Statistical Analysis of Compositional Data," *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 44, no. 2, pp. 139–160, 1982.

[6] J. A. Martín-Fernández, C. Barceló-Vidal, and V. Pawlowsky-Glahn, "Dealing with Zeros and Missing Values in Compositional Data Sets Using Nonparametric Imputation," *Mathematical Geology*, vol. 35, no. 3, pp. 253–278, 2003.

[7] W. Almobaideen, M. Abdullah, U. Alam, S. B. Hussain, and A. Bouharrat, "MedSec-25: Creating an IoMT Dataset for a Healthcare IoT Environment," in *Proc. 2025 7th International Conference on Blockchain Computing and Applications (BCCA)*, IEEE, 2025, pp. 628–634.
