# CAS per class vs. binary (normal-vs-attack): feasibility answer + the attack signature

Question posed: **are Clause-Activation Signatures (CAS) possible for each
class — and if not, fall back to the two-class normal/attack split and produce
the signature for the attack class.**

**Short answer: per-class CAS is possible on all three datasets and already
exists (verified artifacts, summarized in Sec. 1). The binary fallback was
also fully implemented this session (Sec. 2) — a dedicated 2-class
normal-vs-attack TM per dataset, with the CPSS/STABILIS attack-class
signature, the E[V] certificate, 3-seed evaluation, and decoded
human-readable evidence lists. The two arms tell the same underlying story
(Sec. 3), which is the real research finding.**

## 1. Per-class CAS: it works — status of the existing verified results

Per-class (family-level) CAS via CPSS exists for all three datasets. Verified
numbers (see `RESULTS.md`, `DATASET_COMPARISON.md`, and the
`task_a_results_*.json` artifacts):

| Dataset | Classes | \|S_pos\| per class (seed 42) | E[V] bound range | TM-argmax macro-F1 | CAS (STABILIS) macro-F1 |
|---|---|---|---|---|---|
| IoTM (CICIoMT2024) | 6 (Benign + 5 families) | 29–58 | ≤72–250 | 0.686 | **0.776** (CAS wins) |
| MedSec-25 (no ports) | 5 (Benign + 4 kill-chain stages) | 16–39 | ≤16–28 | 0.902 (3 seeds: 0.899–0.903) | 0.864 (0.849–0.870) |
| WUSTL-EHMS-2020 | 3 (normal + Spoofing + Data Alteration) | 3–7 | ≤8–19 | 0.637 | 0.567 — but Spoofing recall 0.00 → **0.91** |

Per-class caveats that are already documented and stand:
- IoTM per-class numbers are the published nbins=10 run (`RESULTS.md`).
- MedSec per-class numbers above are the **no-ports** re-run (ports were a
  leakage feature — 60.3% of signature clauses used a port literal; see
  `results_medsec_WITH_PORTS_deprecated/VERIFIED_FINDINGS.md`).
- WUSTL per-class signatures are thin (3–7 inculpatory clauses/class) because
  the dataset is small (3,263 test rows; Spoofing prevalence 6%) — but they
  are exactly what repairs the base detector's blind Spoofing class.

## 2. Binary fallback: normal-vs-attack CAS, the attack signature (new)

New pipeline: `scripts/binary_cas.py`. For each dataset: collapse all attack
families to one ATTACK class, train a **dedicated 2-class TM** (same
hyperparameter family as the per-class runs; fresh, self-consistent models —
the older multiclass pickles no longer match the re-binarized data on disk),
then run the identical CPSS machinery with `FAMILIES = [normal, attack]`:
stability-selected inculpatory signature `S_pos` (evidence FOR attack),
exculpatory `S_neg` (stably normal-indicating), Shah–Samworth E[V]
certificate, val/eval split disjoint as in the per-class pipeline. 3 seeds
(42/7/123) per dataset. Aggregation: `scripts/binary_cas_aggregate.py`.

### Results (mean over 3 seeds; attack class)

| Dataset | Attack prev. (test) | TM-argmax attack F1 | **CAS attack F1** | TM AUROC | **CAS AUROC** | CAS attack R / P | \|S_pos\| | E[V] | Cross-seed Kuncheva |
|---|---|---|---|---|---|---|---|---|---|
| IoTM | 0.815 | 0.9118 ± 0.0003 | **0.9280 ± 0.0071** | 0.819 | **0.936** | 0.910 / 0.947 | ~138 | ≤220 | 0.181 |
| MedSec-25 | 0.906 | **0.9968 ± 0.0001** | 0.9876 ± 0.0026 | **0.993** | 0.972 | 0.982 / 0.993 | ~58 | ≤61 | 0.080 |
| WUSTL-EHMS | 0.125 | **0.5657 ± 0.0031** | 0.5121 ± 0.0247 | 0.706 | **0.820** | 0.521 / 0.518 | ~14 | ≤13 | −0.014 |

Raw artifacts: `results{,_medsec,_wustl}/binary_cas_results_seed{42,7,123}_thr0.6.json`,
`binary_cas_decoded_*.json`, `binary_tm_model_seed*.pkl`,
`results/binary_cas_aggregate.json`.

### The attack signatures, decoded (seed 42; full lists in the decoded JSONs)

- **IoTM** (151 inculpatory clauses): high-rate / SYN-flag / protocol-type
  conjunctions — e.g. `IF Rate>=bin4` (π̂=1.0), `IF syn_flag_number>=bin1`,
  `IF Protocol Type>=bin1 AND ack_flag_number>=bin1 AND NOT(Header_Length>=bin2)
  AND NOT(Rate>=bin2) AND NOT(psh_flag_number>=bin4)`. Exculpatory clauses are
  long low-rate benign-profile conjunctions. Forensically consistent with the
  DDoS/DoS-flooding-dominated attack mix.
- **MedSec-25** (67 inculpatory clauses): flow-timing structure — e.g.
  `IF Flow IAT Max>=bin5 AND Fwd Pkts/s>=bin2 AND NOT(Flow Duration>=bin17) ...`,
  `IF Init Bwd Win Byts>=bin3 AND NOT(Flow IAT Min>=bin15)`. No port literals
  (ports are dropped from this data prep — the leakage fix held).
- **WUSTL-EHMS** (12 inculpatory clauses): destination jitter / inter-packet
  timing / duration, intertwined with physiological vitals — e.g.
  `IF DstJitter>=bin9 AND NOT(Load>=bin1)`, `IF DIntPkt>=bin9 AND Pulse_Rate>=bin1
  AND NOT(SrcLoad>=bin1) ... AND NOT(Resp_Rate>=bin6)`, exculpatory
  `IF NOT(DstJitter>=bin4)`. This is the patient-monitoring attack fingerprint:
  anomalous network timing *coupled with* vital-sign readings.

## 3. What the two arms together actually show (the finding)

**F1 — Per-class CAS is feasible everywhere; the binary arm is a strict
superset fallback, not a replacement.** Per-class signatures exist and are
usable on all three datasets. The binary arm answers a different operational
question (detection, not family attribution) and is what you want when family
labels are too few/unreliable, or when a single auditable attack-evidence
list is the deliverable.

**F2 — The sign of CAS's value-add is governed by base-detector slack, in
both arms identically.** Per-class: CAS beat TM-argmax only on IoTM
(+0.089 macro-F1), lost on MedSec/WUSTL. Binary: CAS wins on IoTM
(0.928 vs 0.912 F1; AUROC 0.936 vs 0.819 — the binary TM over-predicts attack,
recall 0.988 at precision 0.847, and the CAS signature rebalances to
0.910/0.947), loses slightly on MedSec (0.988 vs 0.997 — nothing left to
repair), and on WUSTL trades F1 for a large ranking-quality gain
(AUROC 0.820 vs 0.706; recall 0.52 vs 0.40 at much lower precision).
This replicates, at binary granularity, the DATASET_COMPARISON law: *the
signature layer is a denoiser/rebalancer when the base detector is weak or
miscalibrated per-class, and lossy compression when the base is already at
ceiling.*

**F3 — The binary attack signature is the most auditable artifact of the
study.** One compact evidence list (12–151 clauses) with per-clause stability
proportions and signs, decodable to plain `IF ... AND ...` feature-threshold
rules (Sec. 2). On WUSTL it surfaces the forensically correct mechanism
(jitter/timing + vitals) from only 3,263 rows.

**F4 — Cross-seed signature stability is still weak in the binary arm**
(Kuncheva: IoTM 0.181, MedSec 0.080, WUSTL −0.014) — better than the
per-class near-zero on IoTM but far from stable. Same root cause documented
in `RESULTS.md` Sec 7: raw clause indices are seed-dependent; stability
claims need clause-content alignment, which remains out of scope. Reported,
not smoothed over.

**F5 — E[V] certificates track pool selectivity, not accuracy.** Tight on
WUSTL (≤13) and MedSec (≤61), loose on IoTM (≤220 against |S|≈180) — same
honest caveat as the per-class run: on IoTM the bound certifies the signature
is much smaller than the pool, not that it is near-noise-free.

## 4. Caveats on this session's data state

- Prior sessions left `booleanized_data.npz` re-binarized *after* some
  multiclass models were trained (IoTM npz is now nbins=5; MedSec npz is
  nbins≈15 no-ports). The saved multiclass pickles do **not** match the
  current npz files (verified by literal-width check). All binary-arm numbers
  here use freshly trained, self-consistent binary TMs on the current npz —
  so binary-vs-per-class footing differs slightly on IoTM (nbins 5 vs 10) and
  MedSec (nbins ~15 vs 10). WUSTL is on identical footing (unchanged npz).
- Train/test attack prevalence differs on IoTM (0.90 train / 0.82 test) and
  WUSTL is heavily imbalanced (0.125) — the WUSTL binary F1s sit on 408
  attack eval rows; treat ±1 seed-std as noise floor.
- pi_thr=0.6, B=15 CPSS subsamples (spec: 50) — same reduction as the
  per-class runs, for tractability.

## 5. Recommendation

Report **per-class CAS as the primary result** (it exists, is verified, and
answers the attribution question), and the **binary attack CAS as the
robustness/operational arm**: one auditable attack evidence list per dataset,
stronger attack ranking than the raw TM on the two weaker detectors
(IoTM AUROC +0.12, WUSTL AUROC +0.11), and the artifact to hand an analyst
when family-level labels cannot be trusted.

## Files added this session

- `scripts/binary_cas.py` — binary TM training + CPSS attack-signature
  extraction + eval + literal decoding, per dataset/seed
- `scripts/binary_cas_aggregate.py` — 3-seed aggregation + cross-seed
  Kuncheva on the attack signature
- `results{,_medsec,_wustl}/binary_cas_*` — models, metrics, decoded
  signatures, logs
- `results/binary_cas_aggregate.json` — cross-seed summary
