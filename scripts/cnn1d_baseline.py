"""
Small 1D-CNN reference point under the partitioned protocol.

The paper's non-TM baselines are all tree or linear models. This adds a compact
convolutional network so the comparison covers a deep-learning reference too.
The 154 Boolean literals are treated as a length-154 sequence with one channel,
which lets the convolutions pick up local structure in the thermometer code
(adjacent bits of one feature are consecutive).

Protocol is identical to the other baselines: fitted on core+cpss, early
stopping and model choice on selm, read once on the full test set, three seeds.
Class-balanced loss, matching the balanced weighting the other baselines use.

    python3 scripts/cnn1d_baseline.py
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from heldout_protocol import load, train_split, OUT, SEEDS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, PATIENCE = 40, 512, 6


class CNN1D(nn.Module):
    """~30k parameters: two conv blocks, global pooling, one hidden layer."""
    def __init__(self, n_bits, n_cls, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2), nn.BatchNorm1d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(width * 2, 64), nn.ReLU(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def run_seed(Xc, yc, Xs, ys, Xt, yt, n_cls, seed, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN1D(Xc.shape[1], n_cls).to(DEV)
    cnt = np.bincount(yc, minlength=n_cls).astype(np.float64)
    w = torch.tensor(len(yc) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xc_t = torch.tensor(Xc, dtype=torch.float32)
    yc_t = torch.tensor(yc, dtype=torch.long)
    Xs_t = torch.tensor(Xs, dtype=torch.float32).to(DEV)
    Xt_t = torch.tensor(Xt, dtype=torch.float32).to(DEV)

    def evaluate(X, y):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(model(X[i:i + 8192]).argmax(1).cpu().numpy())
        return float(f1_score(y, np.concatenate(out), average="macro")), np.concatenate(out)

    best, best_state, bad = -1.0, None, 0
    n = len(yc_t)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb, yb = Xc_t[b].to(DEV), yc_t[b].to(DEV)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        f1s, _ = evaluate(Xs_t, ys)
        if f1s > best:
            best, bad = f1s, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"    early stop at epoch {ep+1}"); break
    model.load_state_dict(best_state)
    f1t, pred = evaluate(Xt_t, yt)
    acc = float((pred == yt).mean())
    n_par = sum(p.numel() for p in model.parameters())
    return best, f1t, acc, n_par


def main():
    def log(s): print(s, flush=True)
    Xtr, ytr, Xte, yte = load()
    core, cpss, selm = train_split(ytr)
    idx = np.sort(np.concatenate([core, cpss]))
    Xc, yc = Xtr[idx].astype(np.float32), ytr[idx]
    Xs, ys = Xtr[selm].astype(np.float32), ytr[selm]
    Xt = Xte.astype(np.float32)
    n_cls = int(np.unique(ytr).size)
    log(f"[data] device={DEV} train={Xc.shape} selm={Xs.shape} test={Xt.shape} classes={n_cls}")

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        sel, f1t, acc, n_par = run_seed(Xc, yc, Xs, ys, Xt, yte, n_cls, seed, log)
        rows.append({"seed": seed, "selm_f1": sel, "test_f1": f1t, "test_acc": acc,
                     "params": n_par})
        log(f"  [cnn1d] seed{seed} selm={sel:.4f} TEST f1={f1t:.4f} acc={acc:.4f} "
            f"params={n_par} ({time.time()-t0:.0f}s)")
    f = [r["test_f1"] for r in rows]; a = [r["test_acc"] for r in rows]
    res = {"cnn1d": {"test_f1_mean": float(np.mean(f)), "test_f1_sd": float(np.std(f)),
                     "test_acc_mean": float(np.mean(a)), "params": rows[0]["params"],
                     "per_seed": rows}}
    json.dump(res, open(f"{OUT}/cnn1d_baseline.json", "w"), indent=2)
    log(f"[done] cnn1d TEST f1={np.mean(f):.4f}+-{np.std(f):.4f} acc={np.mean(a):.4f}")


if __name__ == "__main__":
    main()
