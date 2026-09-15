"""
Decode STABILIS signature clauses back into human-readable literal
conjunctions -- the actual forensic evidence list, not just clause indices.
"""
import json
import pickle
import sys

DATASET_FAMILIES = {
    "medsec": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
    "iotm": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
    "wustl": ["normal", "Spoofing", "Data Alteration"],
}
DATASET_DIRS = {"medsec": "results_medsec", "iotm": "results", "wustl": "results_wustl"}


def decode_clause(tm, class_idx, clause_idx_within_class, literal_names):
    """Return (polarity, list of literal strings) for one clause."""
    n_clauses = tm.clause_banks[class_idx].number_of_clauses
    polarity = "positive" if clause_idx_within_class < n_clauses // 2 else "negative"
    literals = tm.clause_banks[class_idx].get_literals().astype(int)
    n_pos_lits = literals.shape[1] // 2
    row = literals[clause_idx_within_class]
    true_form = [literal_names[i] for i in range(n_pos_lits) if row[i]]
    neg_form = [f"NOT({literal_names[i]})" for i in range(n_pos_lits) if row[n_pos_lits + i]]
    return polarity, true_form + neg_form


def main(dataset="medsec", seed=42, pi_thr=0.6):
    families = DATASET_FAMILIES[dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{DATASET_DIRS[dataset]}"

    with open(f"{base}/tm_model_seed{seed}.pkl", "rb") as f:
        tm = pickle.load(f)
    with open(f"{base}/literal_names.json") as f:
        lit_data = json.load(f)
    literal_names = lit_data["literal_names"]
    with open(f"{base}/stabilis_signatures_seed{seed}_thr{pi_thr}.json") as f:
        sig = json.load(f)

    n_clauses_per_class = tm.number_of_clauses
    out_lines = [f"# Clause-Activation Signatures: {dataset} (seed {seed}, pi_thr={pi_thr})\n"]

    report = {}
    for fam in families:
        sig_entry = sig["signatures"][fam]
        S_pos = sig_entry["pos"] if isinstance(sig_entry, dict) else sig_entry
        S_neg = sig_entry.get("neg", []) if isinstance(sig_entry, dict) else []
        pi_hat_all = sig["diagnostics"][fam]["pi_hat"]
        mean_coef_all = sig["diagnostics"][fam]["mean_coef"]

        out_lines.append(f"\n## Family: {fam}  ({len(S_pos)} inculpatory + {len(S_neg)} exculpatory signature clauses)\n")
        rows = {"inculpatory": [], "exculpatory": []}
        for role, S in [("inculpatory", S_pos), ("exculpatory", S_neg)]:
            for j in S:
                src_class = j // n_clauses_per_class
                clause_within = j % n_clauses_per_class
                polarity, lits = decode_clause(tm, src_class, clause_within, literal_names)
                rows[role].append({
                    "clause_id": j,
                    "source_class": families[src_class],
                    "own_class": families[src_class] == fam,
                    "polarity": polarity,
                    "pi_hat": round(pi_hat_all[j], 3),
                    "mean_coef": round(mean_coef_all[j], 4),
                    "n_literals": len(lits),
                    "literals": lits,
                })
        rows["inculpatory"].sort(key=lambda r: -r["pi_hat"])
        rows["exculpatory"].sort(key=lambda r: -r["pi_hat"])
        report[fam] = rows
        rows = rows["inculpatory"]  # keep the existing markdown listing focused on inculpatory

        own_class_count = sum(1 for r in rows if r["own_class"])
        out_lines.append(f"({own_class_count}/{len(rows)} clauses drawn from {fam}'s own clause bank; "
                          f"{len(rows)-own_class_count} drawn from other classes' banks)\n")

        out_lines.append("**Inculpatory (evidence FOR this family):**\n")
        for r in rows[:15]:  # cap listing for readability; full list in JSON
            src_tag = "" if r["own_class"] else f"  [from {r['source_class']}'s bank]"
            lit_str = " AND ".join(r["literals"]) if r["literals"] else "(empty clause / always true)"
            out_lines.append(f"- clause #{r['clause_id']} (pi_hat={r['pi_hat']}, coef={r['mean_coef']:+.4f}, "
                              f"{r['polarity']}, {r['n_literals']} literals){src_tag}:\n  IF {lit_str}\n")
        if len(rows) > 15:
            out_lines.append(f"  ... and {len(rows)-15} more (see JSON for full list)\n")

        exc_rows = report[fam]["exculpatory"]
        if exc_rows:
            out_lines.append(f"\n**Exculpatory (evidence AGAINST this family, {len(exc_rows)} clauses):**\n")
            for r in exc_rows[:8]:
                src_tag = "" if r["own_class"] else f"  [from {r['source_class']}'s bank]"
                lit_str = " AND ".join(r["literals"]) if r["literals"] else "(empty clause / always true)"
                out_lines.append(f"- clause #{r['clause_id']} (pi_hat={r['pi_hat']}, coef={r['mean_coef']:+.4f}, "
                                  f"{r['polarity']}, {r['n_literals']} literals){src_tag}:\n  IF {lit_str}\n")
            if len(exc_rows) > 8:
                out_lines.append(f"  ... and {len(exc_rows)-8} more (see JSON for full list)\n")

    with open(f"{base}/decoded_signatures_seed{seed}_thr{pi_thr}.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(f"{base}/decoded_signatures_seed{seed}_thr{pi_thr}.md", "w") as f:
        f.write("\n".join(out_lines))

    print("\n".join(out_lines))
    print(f"\nsaved full decoded signatures to {base}/decoded_signatures_seed{seed}_thr{pi_thr}.{{json,md}}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "medsec"
    main(dataset)
