"""
Method 3 -- DISJUNCT: cover-free signature design with identifiability
certificates (forensic_fingerprint_methods.md Sec "METHOD 3"), built on top
of Method 1's STABILIS signatures (S_l,pos, the inculpatory clause sets).

Incidence matrix B in {0,1}^(m x L): B[j,l] = 1 iff clause j is in family l's
inculpatory signature S_l,pos. Since L (number of families) is small (5-6),
the t-disjunctness certificate is computed exactly by brute force over all
subsets of the other L-1 columns -- no approximation needed at this scale.
"""
import argparse
import itertools
import json
import sys
import numpy as np

DATASET_FAMILIES = {
    "medsec": ["Benign", "Reconnaissance", "Initial access", "Lateral movement", "Exfiltration"],
    "iotm": ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
    "wustl": ["normal", "Spoofing", "Data Alteration"],
}
DATASET_DIRS = {"medsec": "results_medsec", "iotm": "results", "wustl": "results_wustl"}


def build_incidence_matrix(signatures, families, m):
    """B: (m, L) boolean. Column l = family l's inculpatory clause indices."""
    L = len(families)
    B = np.zeros((m, L), dtype=bool)
    for i, fam in enumerate(families):
        sig = signatures[fam]
        S_pos = sig["pos"] if isinstance(sig, dict) else sig
        B[S_pos, i] = True
    return B


def min_cover_size(B, col):
    """Minimum number of OTHER columns whose union covers column `col`'s
    support. Brute force over subset sizes 1..L-1 (L is small: 5-6), exact.
    Returns L (i.e. "uncoverable within this code") if no subset covers it."""
    L = B.shape[1]
    target = B[:, col]
    if not target.any():
        return 0  # empty signature is trivially "covered" by the empty union
    other_cols = [c for c in range(L) if c != col]
    for size in range(1, len(other_cols) + 1):
        for combo in itertools.combinations(other_cols, size):
            union = np.zeros(B.shape[0], dtype=bool)
            for c in combo:
                union |= B[:, c]
            if (target & ~union).sum() == 0:  # target fully covered
                return size
    return L  # not covered by union of all others


def t_disjunct_certificate(B, families):
    covers = {}
    for i, fam in enumerate(families):
        mc = min_cover_size(B, i)
        covers[fam] = mc
    t_hat = min(covers.values()) - 1
    return t_hat, covers


def greedy_overlap_reduction(B, families, max_iters=200):
    """Heuristic (per spec, "welding-code heuristic", state as heuristic not
    certified): repeatedly remove the clause that is shared by the most
    families' signatures (highest row-sum > 1), to reduce column overlap and
    raise t_hat. Stops when no clause is shared by >1 family or max_iters hit."""
    B = B.copy()
    removed = 0
    for _ in range(max_iters):
        row_sums = B.sum(axis=1)
        shared = np.where(row_sums > 1)[0]
        if len(shared) == 0:
            break
        # remove the most-shared clause from all but its single "best" family
        # (the family where it has the... we don't have per-family strength
        # here without re-joining diagnostics; simplest defensible heuristic:
        # drop it from ALL families it's shared in, since it cannot serve as
        # a *distinguishing* signature clause for any of them while shared).
        j = shared[np.argmax(row_sums[shared])]
        B[j, :] = False
        removed += 1
    return B, removed


def simulate_mixture_decoding(B, families, t=2, n_trials=500, seed=0, noise_p=0.0, comp_threshold=1.0):
    """Synthetic combinatorial validation of Theorems 1/2: for n_trials random
    size-t family subsets, form z = OR of their B columns, optionally flip
    each bit independently w.p. noise_p, and decode.

    comp_threshold=1.0 is strict COMP (spec's Theorem 1 decoder, exact and
    correct only in the NOISELESS case): family l decoded present iff ALL of
    B[:,l]'s clauses are in z. comp_threshold<1.0 is thresholded-COMP (spec's
    explicit noisy-case decoder, Theorem 2): family l decoded present iff at
    least that FRACTION of its signature clauses are in z. The spec states
    plainly that strict COMP is "the wrong noisy decoder" -- this lets us
    show that honestly rather than only reporting strict COMP's noise
    failure without the decoder noise itself calls for.
    """
    rng = np.random.RandomState(seed)
    L = len(families)
    m = B.shape[0]
    correct = 0
    results = []
    for _ in range(n_trials):
        active = tuple(sorted(rng.choice(L, size=t, replace=False)))
        z = np.zeros(m, dtype=bool)
        for c in active:
            z |= B[:, c]
        if noise_p > 0:
            flips = rng.random(m) < noise_p
            z = z ^ flips
        decoded = []
        for c in range(L):
            sig = B[:, c]
            n_sig = sig.sum()
            if n_sig == 0:
                continue
            frac_present = (sig & z).sum() / n_sig
            if frac_present >= comp_threshold:
                decoded.append(c)
        decoded = tuple(sorted(decoded))
        is_correct = (decoded == active)
        correct += is_correct
        results.append({"active": [families[c] for c in active],
                          "decoded": [families[c] for c in decoded],
                          "correct": bool(is_correct)})
    return correct / n_trials, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASET_FAMILIES.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi_thr", type=float, default=0.6)
    args = ap.parse_args()

    families = DATASET_FAMILIES[args.dataset]
    base = f"/FPTM/CAS_IoMT_Empirical/{DATASET_DIRS[args.dataset]}"

    with open(f"{base}/stabilis_signatures_seed{args.seed}_thr{args.pi_thr}.json") as f:
        sig_data = json.load(f)
    signatures = sig_data["signatures"]
    m = len(sig_data["diagnostics"][families[0]]["pi_hat"])

    B = build_incidence_matrix(signatures, families, m)
    sig_sizes = {fam: int(B[:, i].sum()) for i, fam in enumerate(families)}
    print("signature (inculpatory) sizes per family:", sig_sizes)

    t_hat_raw, covers_raw = t_disjunct_certificate(B, families)
    print(f"\nRAW pool: t_hat = {t_hat_raw}  (min-cover per family: {covers_raw})")

    B_reduced, n_removed = greedy_overlap_reduction(B, families)
    t_hat_reduced, covers_reduced = t_disjunct_certificate(B_reduced, families)
    reduced_sizes = {fam: int(B_reduced[:, i].sum()) for i, fam in enumerate(families)}
    print(f"AFTER greedy overlap reduction ({n_removed} clauses dropped): "
          f"t_hat = {t_hat_reduced}  (min-cover per family: {covers_reduced})")
    print("reduced signature sizes:", reduced_sizes)

    # simulate mixture decoding at t=2 on both raw and reduced B, noiseless and with noise
    out = {"dataset": args.dataset, "seed": args.seed,
           "signature_sizes": sig_sizes, "t_hat_raw": t_hat_raw, "covers_raw": covers_raw,
           "t_hat_reduced": t_hat_reduced, "covers_reduced": covers_reduced,
           "n_clauses_removed": n_removed, "reduced_signature_sizes": reduced_sizes}

    for tag, Bmat in [("raw", B), ("reduced", B_reduced)]:
        for noise_p, thr in [(0.0, 1.0), (0.05, 1.0), (0.05, 0.9), (0.05, 0.8), (0.15, 0.8)]:
            acc, _ = simulate_mixture_decoding(Bmat, families, t=2, n_trials=500, seed=args.seed,
                                                 noise_p=noise_p, comp_threshold=thr)
            decoder = "strict-COMP" if thr == 1.0 else f"thresholded-COMP@{thr}"
            print(f"[{tag}] t=2 mixture decoding accuracy, {decoder}, noise_p={noise_p}: {acc:.4f}")
            out[f"mixture_decode_acc_t2_{tag}_noise{noise_p}_thr{thr}"] = acc

    with open(f"{base}/disjunct_results_seed{args.seed}_thr{args.pi_thr}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved DISJUNCT results.")


if __name__ == "__main__":
    main()
