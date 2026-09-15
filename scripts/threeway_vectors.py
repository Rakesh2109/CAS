"""Save union-over-grid CPSS (pi_hat, mean_coef) at B=15 on the fit half,
full 2400-clause pool -- identical half-fits to threeway_kappa.py."""
import sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from threeway_kappa import cpss_perkappa, splits, BASE, SEEDS, NCLS

def job(a):
    seed, fi = a
    t0 = time.time()
    yte = np.load(f"{BASE}/booleanized_data.npz")["yte"]
    Z = np.load(f"{BASE}/Zte_seed{seed}.npy", mmap_mode="r")
    fit, _, _ = splits(Z.shape[0])
    Zf = np.asarray(Z[fit], dtype=np.float32)
    _, (upi, ucf) = cpss_perkappa(Zf, (yte[fit] == fi).astype(int), seed=seed + fi)
    print(f"[v] seed{seed} fam{fi} {time.time()-t0:.0f}s", flush=True)
    return (seed, fi, upi, ucf)

if __name__ == "__main__":
    with Pool(processes=18) as p:
        res = p.map(job, [(s, f) for s in SEEDS for f in range(NCLS)])
    out = {}
    for s, f, upi, ucf in res:
        out[f"pi_{s}_{f}"] = upi; out[f"cf_{s}_{f}"] = ucf
    np.savez_compressed(f"{BASE}/threeway_vectors.npz", **out)
    print("[done] threeway_vectors.npz", flush=True)
