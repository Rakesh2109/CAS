"""Check paper aggregates and reconstruct full-pool CAS from saved vectors."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
data = json.loads((R / "threeway_cas.json").read_text())
vectors = np.load(R / "threeway_vectors.npz")
labels = np.load(R / "booleanized_data.npz")["yte"]
perm = np.random.RandomState(0).permutation(len(labels))
fit, sel, rep = np.split(perm, [len(labels)//2, len(labels)//2+len(labels)//4])
assert len(set(fit) & set(sel)) == len(set(fit) & set(rep)) == len(set(sel) & set(rep)) == 0
out = {"n": {"fit": len(fit), "sel": len(sel), "rep": len(rep)}, "runs": []}
for pool in data["pools"].values():
    best = max(pool["grid"], key=lambda key: pool["grid"][key]["sel_f1"])
    assert best == f"{pool['selected_B']}|{pool['selected_pi_thr']}"
    for row in pool["grid"].values():
        for metric in ("rep_f1", "rep_acc"):
            values = [v[metric] for v in row["per_seed"]]
            np.testing.assert_allclose([np.mean(values), np.std(values)],
                                       [row[metric], row[metric+"_sd"]], atol=1e-12)
for si, seed in enumerate([42, 7, 123]):
    z = np.asarray(np.load(R/f"Zte_seed{seed}.npy", mmap_mode="r")[rep], np.float32)
    w = np.load(R/f"W_seed{seed}.npy")
    base = np.argmax(np.column_stack([z[:,c*400:(c+1)*400] @ w[c*400:(c+1)*400]
                                     for c in range(6)]), axis=1)
    signed, unsigned = np.zeros((len(rep), 6)), np.zeros((len(rep), 6))
    supports, pos, diagnostics = [], [], []
    for c in range(6):
        pi, cf = vectors[f"pi_{seed}_{c}"], vectors[f"cf_{seed}_{c}"]
        support = pi >= .8
        positive, negative = support & (cf > 0), support & (cf < 0)
        supports.append(int(support.sum())); pos.append(int(positive.sum()))
        den = pi[positive].sum()
        if den > 0:
            signed[:,c] = (z[:,positive]@pi[positive]-z[:,negative]@pi[negative])/den
        if pi[support].sum() > 0:
            unsigned[:,c] = z[:,support]@pi[support]/pi[support].sum()
        diagnostics.append({"q_hat": float(pi.sum()),
                            "bound_plugin": float(pi.sum()**2/(.6*len(pi)))})
    expected = data["pools"]["full"]["grid"]["15|0.8"]["per_seed"][si]
    assert pos == expected["S_pos"] and supports == expected["S"]
    cas_pred = np.argmax(signed, axis=1)
    cas_f1 = f1_score(labels[rep], cas_pred, average="macro")
    np.testing.assert_allclose(cas_f1, expected["rep_f1"], atol=1e-12)
    matrix = np.column_stack([(vectors[f"pi_{seed}_{c}"]>=.8)&
                              (vectors[f"cf_{seed}_{c}"]>0) for c in range(6)])
    private = [int((matrix[:,c] & ~np.any(np.delete(matrix,c,axis=1),axis=1)).sum())
               for c in range(6)]
    correct, total = 0, 0
    for size in range(1,6):
        for active in combinations(range(6),size):
            mixture = np.any(matrix[:,active], axis=1)
            prediction = tuple(np.where(np.all(~matrix | mixture[:,None], axis=0))[0])
            correct += prediction == active; total += 1
    out["runs"].append({"seed":seed,"base_f1":float(f1_score(labels[rep],base,average="macro")),
        "signed_f1":float(cas_f1),"unsigned_f1":float(f1_score(labels[rep],np.argmax(unsigned,1),average="macro")),
        "S":supports,"S_pos":pos,"diagnostics":diagnostics,"private_clauses":private,
        "exact_mixtures":correct,"total_mixtures":total,
        "base_classes":classification_report(labels[rep],base,output_dict=True),
        "cas_classes":classification_report(labels[rep],cas_pred,output_dict=True)})
out["mean"] = {key:float(np.mean([r[key] for r in out["runs"]]))
               for key in ("base_f1","signed_f1","unsigned_f1")}
out["sd"] = {key:float(np.std([r[key] for r in out["runs"]]))
             for key in ("base_f1","signed_f1","unsigned_f1")}
out["pruning"] = {k:data[k] for k in ("kstar_removed_per_class","keep_per_class","pool_pruned")}
sel_curve = np.mean(data["prune_curve_sel"],axis=0)
valid = np.where(sel_curve >= sel_curve[0]-.01)[0]
assert data["ks"][valid[-1]] == data["kstar_removed_per_class"]
binary = json.loads((R/"threeway_binary.json").read_text())
best = max(binary["grid"], key=lambda key: binary["grid"][key]["sel_macro_f1"])
assert best == f"{binary['selected_B']}|{binary['selected_pi_thr']}"
binary_runs = []
for seed in [42,7,123]:
    z = np.asarray(np.load(R/f"Zte_binary_seed{seed}.npy",mmap_mode="r")[rep],np.float32)
    w = np.load(R/f"W_binary_seed{seed}.npy")
    cs = np.column_stack([z[:,c*400:(c+1)*400] @ w[c*400:(c+1)*400] for c in range(2)])
    y = labels[rep] != 0
    binary_runs.append({"seed":seed,
        "attack_f1":float(f1_score(y,np.argmax(cs,axis=1))),
        "clipped_auroc":float(roc_auc_score(y,np.clip(cs[:,1],0,320)/320)),
        "raw_auroc":float(roc_auc_score(y,cs[:,1])),
        "margin_auroc":float(roc_auc_score(y,cs[:,1]-cs[:,0]))})
out["binary"] = {"runs":binary_runs,
                 "mean":{k:float(np.mean([r[k] for r in binary_runs]))
                         for k in ["attack_f1","clipped_auroc","raw_auroc","margin_auroc"]}}
np.testing.assert_allclose(out["binary"]["mean"]["clipped_auroc"],
                           binary["tm_baseline"]["rep"]["auroc"],atol=1e-12)
(ROOT/"paper"/"review"/"results_audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"mean":out["mean"],"sd":out["sd"],
                  "binary":out["binary"]["mean"],
                  "private_clauses":[r["private_clauses"] for r in out["runs"]],
                  "exact_mixtures":[r["exact_mixtures"] for r in out["runs"]]},indent=2))
