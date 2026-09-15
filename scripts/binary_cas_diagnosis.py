"""
Why the binary CAS arm behaves differently from the multiclass arm.

Four checks, all on cached activations/weights and the same fit/sel/rep split:
  D1  Binary TM baseline is prior-collapsed (Benign recall, per-family rates).
  D2  How much of the CAS "gain" is decision-threshold calibration: tune one
      scalar threshold on the TM class-sum margin using the selection quarter,
      then read the report quarter.
  D3  Does magnitude pruning keep the clauses CAS actually uses (binary vs
      multiclass), and what fraction of the pool is retained.
  D4  Does the epsilon non-degradation pruning rule ever bind in the binary
      case (base macro-F1 vs number of removed clauses).
  D5  Benign-class covariate shift between the official Train.csv and Test.csv
      splits: per-class thermometer-bit profile distance, the most-shifted
      literals, and the same 6-class model's Benign recall for contrast.

Output: results/binary_cas_diagnosis.json
"""
import json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

BASE = "/FPTM/CAS_IoMT_Empirical/results"
SEEDS = [42, 7, 123]
NC = 400
FAM = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]


def splits(n):
    p = np.random.RandomState(0).permutation(n)
    h, q = n // 2, n // 4
    return p[:h], p[h:h + q], p[h + q:]


def class_sums(Z, W, ncls, nc=NC):
    return np.column_stack([Z[:, c * nc:(c + 1) * nc] @ W[c * nc:(c + 1) * nc]
                            for c in range(ncls)])


def main():
    d = np.load(f"{BASE}/booleanized_data.npz")
    yte = d["yte"]; yb = (yte != 0).astype(int)
    fit, sel, rep = splits(len(yb))
    out = {"split": {"fit": len(fit), "sel": len(sel), "rep": len(rep)},
           "train_attack_prevalence": float((d["ytr"] != 0).mean()),
           "test_attack_prevalence": float(yb.mean())}

    d1, d2, d3b = [], [], []
    for s in SEEDS:
        Z = np.load(f"{BASE}/Zte_binary_seed{s}.npy", mmap_mode="r")
        W = np.load(f"{BASE}/W_binary_seed{s}.npy")
        cs_sel = class_sums(np.asarray(Z[sel], np.float32), W, 2)
        cs_rep = class_sums(np.asarray(Z[rep], np.float32), W, 2)

        # D1
        pred = np.argmax(cs_rep, 1)
        cm = confusion_matrix(yb[rep], pred)
        d1.append({"seed": s,
                   "macro_f1": float(f1_score(yb[rep], pred, average="macro")),
                   "benign_recall": float(cm[0, 0] / cm[0].sum()),
                   "confusion": cm.tolist(),
                   "per_family_pred_attack_rate":
                       {f: float((pred[yte[rep] == i] == 1).mean()) for i, f in enumerate(FAM)},
                   "mean_class_sums_on_benign": cs_rep[yb[rep] == 0].mean(0).tolist()})

        # D2
        m_sel, m_rep = cs_sel[:, 1] - cs_sel[:, 0], cs_rep[:, 1] - cs_rep[:, 0]
        cand = np.quantile(m_sel, np.linspace(0.001, 0.999, 400))
        fs = [f1_score(yb[sel], (m_sel > t).astype(int), average="macro") for t in cand]
        t = float(cand[int(np.argmax(fs))])
        d2.append({"seed": s, "threshold": t,
                   "argmax_rep_macro_f1": float(f1_score(yb[rep], (m_rep > 0).astype(int), average="macro")),
                   "tuned_sel_macro_f1": float(max(fs)),
                   "tuned_rep_macro_f1": float(f1_score(yb[rep], (m_rep > t).astype(int), average="macro")),
                   "margin_rep_auroc": float(roc_auc_score(yb[rep], m_rep))})

        # D3 binary
        keep = set()
        for c in (2 * [0])[:0] or (0, 1):
            o = np.argsort(np.abs(W[c * NC:(c + 1) * NC]), kind="stable")
            keep |= set((c * NC + o[NC - 15:]).tolist())
        tot = surv = 0
        for c in (0, 1):
            snap = json.load(open(f"{BASE}/binary_revision_snaps_full_seed{s}_class{c}.json"))
            pi = np.array(snap["25"][0])
            S = set(np.where(pi >= 0.9)[0].tolist())
            tot += len(S); surv += len(S & keep)
        d3b.append({"seed": s, "cas_supports": tot, "survive_pruning": surv,
                    "survival_rate": surv / tot, "pool_retention": 30 / 800})

    # D3 multiclass contrast
    z = np.load(f"{BASE}/threeway_vectors.npz")
    d3m = []
    for s in SEEDS:
        W = np.load(f"{BASE}/W_seed{s}.npy"); keep = set()
        for c in range(6):
            o = np.argsort(np.abs(W[c * NC:(c + 1) * NC]), kind="stable")
            keep |= set((c * NC + o[NC - 65:]).tolist())
        tot = surv = 0
        for c in range(6):
            S = set(np.where(z[f"pi_{s}_{c}"] >= 0.8)[0].tolist())
            tot += len(S); surv += len(S & keep)
        d3m.append({"seed": s, "cas_supports": tot, "survive_pruning": surv,
                    "survival_rate": surv / tot, "pool_retention": 390 / 2400})

    # D4
    d4 = {}
    for name, f in (("binary", "threeway_binary_pruned.json"), ("multiclass", "threeway_cas.json")):
        j = json.load(open(f"{BASE}/{f}"))
        ks = np.array(j["ks"]); curve = np.array(j["prune_curve_rep"]).mean(0)
        kstar = j["kstar_removed_per_class"]
        d4[name] = {"k0_macro_f1": float(curve[0]), "best_k": int(ks[curve.argmax()]),
                    "best_macro_f1": float(curve.max()), "kstar": int(kstar),
                    "kstar_macro_f1": float(curve[list(ks).index(kstar)]),
                    "monotone_nondecreasing_to_kstar":
                        bool(curve[list(ks).index(kstar)] >= curve[0]),
                    "pool_kept": j["pool_pruned"]}

    # D5 train/test covariate shift
    lit = json.load(open(f"{BASE}/full_bool_nbins10_literals.json"))["literal_names"]
    Xtr, ytr, Xte = d["Xtr"], d["ytr"], d["Xte"]
    prof = {}
    for i, f in enumerate(FAM):
        a, b = Xtr[ytr == i].mean(0), Xte[yte == i].mean(0)
        prof[f] = {"mean_abs_bit_shift": float(np.abs(a - b).mean()),
                   "max_abs_bit_shift": float(np.abs(a - b).max())}
    a, b = Xtr[ytr == 0].mean(0), Xte[yte == 0].mean(0)
    aa, bb = Xtr[ytr != 0].mean(0), Xte[yte != 0].mean(0)
    top = np.argsort(-np.abs(a - b))[:8]
    d5 = {"per_class_profile_shift": prof,
          "top_shifted_benign_literals": [
              {"literal": lit[j], "benign_train": float(a[j]), "benign_test": float(b[j]),
               "attack_train": float(aa[j]), "attack_test": float(bb[j])} for j in top]}
    Z = np.load(f"{BASE}/Zte_seed42.npy", mmap_mode="r")
    W = np.load(f"{BASE}/W_seed42.npy")
    cs6 = class_sums(np.asarray(Z[rep], np.float32), W, 6)
    p6 = np.argmax(cs6, 1)
    d5["multiclass_seed42_benign_recall"] = float((p6[yte[rep] == 0] == 0).mean())
    d5["multiclass_seed42_binarised_macro_f1"] = float(
        f1_score((yte[rep] != 0).astype(int), (p6 != 0).astype(int), average="macro"))

    out.update({"D5_train_test_shift": d5, "D1_baseline_collapse": d1, "D2_threshold_calibration": d2,
                "D3_pruning_overlap": {"binary": d3b, "multiclass": d3m},
                "D4_epsilon_rule": d4})
    cas = {r["seed"]: r["rep"]["macro_f1"] for r in
           json.load(open(f"{BASE}/threeway_binary.json"))["reported"]["per_seed"]}
    diffs = [cas[r["seed"]] - r["tuned_rep_macro_f1"] for r in d2]
    out["summary"] = {
        "tm_argmax_rep_macro_f1": float(np.mean([r["argmax_rep_macro_f1"] for r in d2])),
        "tm_tuned_threshold_rep_macro_f1": float(np.mean([r["tuned_rep_macro_f1"] for r in d2])),
        "tm_tuned_threshold_sd": float(np.std([r["tuned_rep_macro_f1"] for r in d2])),
        "cas_full_rep_macro_f1": float(np.mean(list(cas.values()))),
        "cas_minus_tuned_threshold": float(np.mean(diffs)),
        "cas_minus_tuned_threshold_sd": float(np.std(diffs)),
        "fraction_of_gain_from_calibration":
            float((np.mean([r["tuned_rep_macro_f1"] for r in d2])
                   - np.mean([r["argmax_rep_macro_f1"] for r in d2]))
                  / (np.mean(list(cas.values())) - np.mean([r["argmax_rep_macro_f1"] for r in d2]))),
    }
    with open(f"{BASE}/binary_cas_diagnosis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(json.dumps(out["D4_epsilon_rule"], indent=2))
    print("[done] results/binary_cas_diagnosis.json")


if __name__ == "__main__":
    main()
