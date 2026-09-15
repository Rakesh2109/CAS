# Dataset Selection Justification: why CICIoMT2024 (IoTM) is the primary dataset

Decision under justification: **CICIoMT2024 (WiFi/MQTT branch) as the sole
primary dataset** for the CAS (clause-activation signature) study, with
WUSTL-EHMS-2020 and MedSec-25 demoted to kept-on-disk supporting evidence.
Compiled 2026-08-22 from three literature-research passes (dataset itself,
competing datasets, TM/interpretability literature) plus this repo's own
empirical results. Every external claim carries a source; items we could not
verify first-hand are marked **[unverified]**.

## 1. The selection criteria (from IDS dataset-validity literature)

The field's canonical critiques give the criteria a defensible IDS dataset
must satisfy:

- Sommer & Paxson, "Outside the Closed World" (IEEE S&P 2010): ML-NIDS
  evaluation fails on non-stationary traffic, semantic gaps, and lack of
  appropriate public data.
- Engelen, Rimmer & Joosen (IEEE SPW 2021) and Lanvin et al. (CRiSIS 2022):
  CIC flow tooling can leak artifacts into labels (CICIDS2017 case) —
  flow-extraction provenance must be checkable.
- Arp et al., "Dos and Don'ts of ML in Computer Security" (USENIX Sec 2022):
  sampling bias, label inaccuracy, spurious correlations must be addressed.
- Ring et al.-style survey consensus ("Network Intrusion Datasets: A Survey,
  Limitations, and Recommendations", 2025, arXiv:2502.06688): most public
  NIDS datasets fail several validity criteria.

## 2. CICIoMT2024 against those criteria

Source facts (UNB dataset page; Dadkhah et al., *Internet of Things*
28:101351, 2024 — peer-reviewed, open access):

- **IoMT-specific**: 40 healthcare devices, **25 real** (baby monitors, HR
  armbands, O2 rings, …) + 15 simulated; Wi-Fi, MQTT, BLE; Faraday-caged RF
  isolation.
- **Attack taxonomy fit for per-class CAS**: 18 attack types in 5 families
  (DDoS×4, DoS×4, Recon×4, Spoofing×1=ARP, MQTT×5) + benign — exactly the
  family/subtype structure a clause-signature attribution study needs
  (family level for Task A, subtype level available for LOSO extensions).
- **No IP/port leakage by design**: the published flow schema (~44–46
  CIC-extracted features) contains no IP addresses or ports — the leakage
  channel that Engelen et al. document for CICIDS2017 is absent here.
  (Contrast: MedSec-25's CICFlowMeter schema *does* carry ports — we caught
  60.3% of our early MedSec signature clauses relying on a port literal and
  had to drop them; see `results_medsec_WITH_PORTS_deprecated/VERIFIED_FINDINGS.md`.)
- **Adoption**: ~228 citations (Scopus snapshot, late 2025) and growing;
  de-facto IoMT benchmark in 2025–26 papers; IEEE DataPort mirror.
- **Adversarial scrutiny exists and is citable**: Doménech Fons (*Internet of
  Things*, 2025) critiques its windowing/splitting/temporal-dependence
  choices and shows up to 66.9% F1 drop under cross-dataset shift — a
  published critique we can cite and explicitly address rather than be
  surprised by.
- Known quirks to disclose: extreme imbalance (DDoS+DoS ≈ 95.7% of flows,
  Ping Sweep <0.01% — hence our stratified caps and macro-F1 reporting);
  `Rate`/`Srate` duplicate; near-constant `Weight`; 15/40 devices simulated.

## 3. Why the alternatives were demoted

| Dataset | Verdict | Reason |
|---|---|---|
| **WUSTL-EHMS-2020** | Supporting only | Real medical testbed with unique physiological+network features, but tiny (~16K rows, 12.5% attack), only 2 MITM attack types, near-saturated in literature (XGBoost F1≈0.81). Our own run: per-class signatures only 3–7 clauses; binary attack CAS AUROC 0.820 but F1 0.51. Great stress-test, not primary evidence. |
| **MedSec-25** | Supporting only | Real capture, MITRE ATT&CK kill-chain labels (beautiful conceptual fit), but: BCCA-2025 conference paper + Kaggle distribution, ~1 year old, thin independent usage, near-saturated benchmark (TM macro-F1 97.8% published), and CICFlowMeter schema carries port leakage (we verified this empirically). Our no-ports rerun dropped TM macro-F1 from ~0.95 to ~0.90. |
| **CICIoT2023** | Companion at most | The standard generic-IoT benchmark (1,500+ citations, 105 real devices, 33 attacks) but **not medical**, flood-dominated (benign ≈2.3%), and near-trivially separable — no room to show a signature method's value. |
| **X-IIoTID** | No (primary) | Kill-chain staged but **industrial** IIoT, not IoMT. |
| **TON_IoT** | No | ~80% attack share (inverted vs reality), heterogeneous sub-split schemas, testbed-synthetic. |
| **Bot-IoT** | No | >90% attack, simulated benign, burst-executed attacks — textbook benchmark-validity failure. |
| **IoT-23** | No | Botnet-C&C-centric, only 3 benign scenarios. |
| **UNSW-NB15** | No | 2015, enterprise (not IoT), documented train/test distribution mismatch and class overlap. |

## 4. The TM-literature angle: CICIoMT2024 is where the TM-IDS baseline lives

- Jaiswal et al. (UiA/CAIR), "A Tsetlin Machine-driven IDS for Next-Gen IoMT
  Security" (arXiv:2604.03205; IEEE SVCC 2026): TM on CICIoMT2024 — 99.5%
  binary (BLE), 90.7% 6-class (WiFi+MQTT), beating DT/RF/XGBoost/LGBM/KNN/NN.
  Uses SMOTE + KBinsDiscretizer — the same toolchain family as ours, so it is
  a directly comparable baseline. **Two caveats we must handle better than
  they did**: (a) F1 averaging method is never stated (their
  precision>accuracy pattern suggests weighted, inflating the headline on
  this imbalanced data); (b) their clause/T/s hyperparameter tables are
  image-only **[unverified]** — we report ours explicitly (400 clauses,
  T=320, s=8.0, 25 epochs) and report macro-F1 with per-class breakdowns.
- The same group's MedSec-25 TM paper (arXiv:2605.16707) reports macro-F1
  explicitly (97.83%) — confirming the field's TM-IDS attention is on IoMT
  datasets right now, and that macro-F1 discipline is a differentiator.
- Earlier TM-IDS work sits on obsolete data (KDD'99/NSL-KDD: Abeyrathna et
  al. 2020) or non-network tasks. No published TM paper on CICIDS2017 /
  CICIoT2023 / UNSW-NB15 was found.

## 5. The novelty slot this selection opens

- IDS explainability is dominated by post-hoc SHAP/LIME on black boxes;
  stability-vetted **rule sets as per-attack-family forensic signatures**
  were not found anywhere in the searched literature.
- CPSS (Meinshausen & Bühlmann 2010; Shah & Samworth 2013) is a
  gold-standard error-controlled selection method with **no prior use on
  Tsetlin Machines and no clause-level use in security** found — a genuine
  methodological-transfer contribution (frame as transfer; the S&S
  guarantees are proven for variable selection, not TM clause mining).
- No TM-based open-set IDS paper exists; our Task B (LOFO + conformal) on
  CICIoMT2024 occupies empty space.

## 6. What our own empirical results say (consistent with the literature)

- IoTM is the **only** dataset where CAS beat the TM baseline — per-class
  (+0.089 macro-F1) and binary (+0.016 attack-F1, +0.12 AUROC) — because its
  base detector has slack; MedSec was already at ceiling. A signature method
  needs a non-saturated base to demonstrate value: IoTM provides exactly
  that (our TM macro-F1 0.686; published CNN ~99% accuracy is
  weighted-inflated on 95.7% flood traffic).
- The structural difficulties we measured (DDoS/DoS mutual confusion; Benign
  recall weakness) are documented dataset-inherent properties in the
  literature, not pipeline bugs.

## 7. Conclusion

CICIoMT2024 is the defensible primary: the only peer-reviewed, widely
adopted, IoMT-specific, multi-protocol benchmark with a no-port-leakage
feature schema, a family/subtype taxonomy that per-class CAS can exploit, a
published TM baseline to beat on *methodological rigor* (macro-F1, verified
hyperparameters, stability selection) rather than headline accuracy, and a
published critique (Doménech 2025) to answer. WUSTL-EHMS-2020 and MedSec-25
remain on disk as supporting stress-tests, not primary evidence.

### Key sources
- UNB CIC dataset page: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
- Dadkhah et al. 2024 (dataset paper): https://www.sciencedirect.com/science/article/pii/S2542660524002920
- Doménech 2025 (critique): https://www.sciencedirect.com/science/article/pii/S2542660525001453
- Jaiswal et al. 2026 (TM baseline): https://arxiv.org/abs/2604.03205
- Jaiswal et al. 2026 (TM on MedSec-25): https://arxiv.org/html/2605.16707v2
- Abeyrathna et al. 2020 (TM-IDS rules): https://ieeexplore.ieee.org/document/9308206/
- Shah & Samworth 2013 (CPSS); Meinshausen & Bühlmann 2010 (stability selection)
- Sommer & Paxson 2010; Engelen et al. 2021 (IEEE SPW); Arp et al. 2022 (USENIX Sec); arXiv:2502.06688 (dataset survey)
- WUSTL-EHMS-2020: Hady et al., IEEE Access 2020; MedSec-25: BCCA 2025 (Kaggle dist.)
