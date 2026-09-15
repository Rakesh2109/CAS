"""Compute binary table metrics and verify full-pool reproduction from cached snapshots."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from threeway_binary_pruned import splits, prep, keep_cols, class_sums, NC, NCLS, SEEDS
from threeway_binary import score_matrix

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results"

def metrics(y, scores):
    pred = scores.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((y == pred).mean()),
            "attack_f1": float(f1_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, scores[:, 1]))}

def aggregate(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows]))} for k in rows[0]}

def main():
    exp = json.loads((BASE / "threeway_binary_pruned.json").read_text())
    old = json.loads((BASE / "threeway_binary.json").read_text())
    y = prep()
    _, _, rep = splits(len(y))
    out = {"seeds": SEEDS, "sd_ddof": 0, "n_report": len(rep),
           "report_attack_count": int(y[rep].sum()), "pools": {},
           "signature_settings": {"seed": 42, "B": 15, "pi_thr": .8}}
    for pool in ("full", "pruned"):
        info = exp["pools"][pool]
        B, thr = info["selected_B"], info["selected_pi_thr"]
        base, cas = [], []
        for seed in SEEDS:
            Z = np.load(BASE / f"Zte_binary_seed{seed}.npy", mmap_mode="r")
            W = np.load(BASE / f"W_binary_seed{seed}.npy")
            cols = np.arange(NC * NCLS) if pool == "full" else keep_cols(W, exp["keep_per_class"])
            Zr = np.asarray(Z[rep][:, cols], dtype=np.float32)
            scores = class_sums(Zr, W[cols], info["clauses_per_class"])
            base.append(metrics(y[rep], scores))
            snaps = [json.loads((BASE / f"binary_revision_snaps_{pool}_seed{seed}_class{c}.json").read_text()) for c in range(NCLS)]
            pis = [np.array(s[str(B)][0]) for s in snaps]
            cfs = [np.array(s[str(B)][1]) for s in snaps]
            cm = metrics(y[rep], score_matrix(Zr, pis, cfs, thr))
            cas.append(cm)
            expected = info["reported"]["per_seed"][SEEDS.index(seed)]
            assert abs(cm["macro_f1"] - expected["rep_f1"]) < 1e-12
            assert abs(cm["accuracy"] - expected["rep_acc"]) < 1e-12
        match = exp["pools"][pool]["grid"]["15|0.8"]["per_seed"][0]
        out["pools"][pool] = {"pool_size": info["pool"], "B": B, "pi_thr": thr,
            "tm": aggregate(base), "cas": aggregate(cas),
            "signature_size": {c: {"total": match["S"][i], "positive": match["S_pos"][i]} for i, c in enumerate(("Benign", "Attack"))},
            "per_seed_tm": base, "per_seed_cas": cas}
    diffs = {key: exp["pools"]["full"]["grid"][key]["rep_f1"] - val["rep"]["macro_f1"] for key, val in old["grid"].items()}
    out["full_pool_max_reproduction_difference"] = max(abs(v) for v in diffs.values())
    assert out["full_pool_max_reproduction_difference"] < .00001
    dest = ROOT / "paper/review/binary_revision/results_audit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
