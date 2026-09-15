"""
Binarized Neural Network reference point under the partitioned protocol.

A BNN is the closest neural analogue to a Tsetlin Machine on the axis the TM is
usually argued on: both reduce inference to bitwise operations over Boolean
inputs, with no floating-point multiply in the deployed model. Weights and
activations are binarized with sign(), gradients flow through a straight-through
estimator, and real-valued latent weights are kept only for training and clipped
to [-1, 1].

Protocol matches the other baselines exactly: fitted on core+cpss, early
stopping and checkpoint choice on selm, read once on the full test set, three
seeds, class-balanced loss.

    python3 scripts/bnn_baseline.py
"""
import sys
# torch 2.12's dynamo expects sys.get_int_max_str_digits, absent from this
# Python 3.11.0rc1 build; provide signature-compatible stand-ins before import.
if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int: return 4300
    def set_int_max_str_digits(maxdigits: "int") -> "None": return None
    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits

import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, "/FPTM/CAS_IoMT_Empirical/scripts")
from heldout_protocol import load, train_split, OUT, SEEDS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_raw():
    """The 22 continuous features, in the same row order as the booleanized
    arrays (verified against their labels before use)."""
    from data_prep import FEATURE_COLS, LABEL_COL, FAMILIES, load_and_sample, clean_inf
    tr, te = load_and_sample()
    tr, te = clean_inf(tr, te)
    f = lambda d: (d[FEATURE_COLS].to_numpy(np.float64).astype(np.float32),
                   d[LABEL_COL].map({x: i for i, x in enumerate(FAMILIES)}).to_numpy(int))
    return f(tr), f(te)
EPOCHS, BATCH, PATIENCE = 60, 512, 8


class SignSTE(torch.autograd.Function):
    """sign() forward, identity-within-|x|<=1 backward (Hubara et al.)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


binarize = SignSTE.apply


class BinLinear(nn.Module):
    """Linear layer whose weights are binarized in the forward pass."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in).uniform_(-1, 1) * 0.1)

    def forward(self, x):
        return torch.nn.functional.linear(x, binarize(self.weight))

    @torch.no_grad()
    def clip_(self):
        self.weight.clamp_(-1, 1)


class BNN(nn.Module):
    def __init__(self, n_bits, n_cls, width=256, binary_input=True):
        super().__init__()
        self.binary_input = binary_input
        self.l1, self.b1 = BinLinear(n_bits, width), nn.BatchNorm1d(width)
        self.l2, self.b2 = BinLinear(width, width), nn.BatchNorm1d(width)
        self.l3, self.b3 = BinLinear(width, n_cls), nn.BatchNorm1d(n_cls)

    def forward(self, x):
        # Boolean inputs map {0,1} -> {-1,+1}; continuous inputs stay real-valued
        # into the first layer, whose weights are still binary (Hubara et al.).
        if self.binary_input:
            x = binarize(x * 2 - 1)
        x = binarize(self.b1(self.l1(x)))
        x = binarize(self.b2(self.l2(x)))
        return self.b3(self.l3(x))         # logits, no binarization on output

    def clip_(self):
        for m in (self.l1, self.l2, self.l3):
            m.clip_()


def run_seed(Xc, yc, Xs, ys, Xt, yt, n_cls, seed, log, binary_input=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BNN(Xc.shape[1], n_cls, binary_input=binary_input).to(DEV)
    cnt = np.bincount(yc, minlength=n_cls).astype(np.float64)
    w = torch.tensor(len(yc) / (n_cls * np.maximum(cnt, 1)),
                     dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xc_t = torch.tensor(Xc, dtype=torch.float32)
    yc_t = torch.tensor(yc, dtype=torch.long)
    Xs_t = torch.tensor(Xs, dtype=torch.float32).to(DEV)
    Xt_t = torch.tensor(Xt, dtype=torch.float32).to(DEV)

    def evaluate(X, y):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(model(X[i:i + 8192]).argmax(1).cpu().numpy())
        p = np.concatenate(out)
        return float(f1_score(y, p, average="macro")), p

    best, best_state, bad = -1.0, None, 0
    n = len(yc_t)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb, yb = Xc_t[b].to(DEV), yc_t[b].to(DEV)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
            model.clip_()
        sched.step()
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
    n_bin = sum(m.weight.numel() for m in (model.l1, model.l2, model.l3))
    return best, f1t, acc, n_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="bool", choices=["bool", "raw"],
                    help="bool: the 154 thermometer bits. raw: the 22 continuous features.")
    a = ap.parse_args()

    def log(s): print(s, flush=True)
    Xtr, ytr, Xte, yte = load()
    if a.input == "raw":
        (Xr_tr, yr_tr), (Xr_te, yr_te) = load_raw()
        assert np.array_equal(yr_tr, ytr), "raw train row order does not match"
        assert np.array_equal(yr_te, yte), "raw test row order does not match"
        Xtr, Xte = Xr_tr, Xr_te
        log("[align] raw features verified against booleanized labels")
    core, cpss, selm = train_split(ytr)
    idx = np.sort(np.concatenate([core, cpss]))
    Xc, yc = Xtr[idx].astype(np.float32), ytr[idx]
    Xs, ys = Xtr[selm].astype(np.float32), ytr[selm]
    Xt = Xte.astype(np.float32)
    if a.input == "raw":                      # z-score using training statistics only
        mu, sd = Xc.mean(0), Xc.std(0)
        sd[sd == 0] = 1.0
        Xc, Xs, Xt = (Xc - mu) / sd, (Xs - mu) / sd, (Xt - mu) / sd
        log("[prep] standardised with train mean/std")
    n_cls = int(np.unique(ytr).size)
    log(f"[data] device={DEV} input={a.input} train={Xc.shape} selm={Xs.shape} "
        f"test={Xt.shape} classes={n_cls}")

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        sel, f1t, acc, n_bin = run_seed(Xc, yc, Xs, ys, Xt, yte, n_cls, seed, log,
                                        binary_input=(a.input == "bool"))
        rows.append({"seed": seed, "selm_f1": sel, "test_f1": f1t, "test_acc": acc,
                     "binary_weights": n_bin})
        log(f"  [bnn-{a.input}] seed{seed} selm={sel:.4f} TEST f1={f1t:.4f} acc={acc:.4f} "
            f"binary_weights={n_bin} ({time.time()-t0:.0f}s)")
    f1_list = [r["test_f1"] for r in rows]
    acc_list = [r["test_acc"] for r in rows]
    res = {f"bnn_{a.input}": {"test_f1_mean": float(np.mean(f1_list)),
                   "test_f1_sd": float(np.std(f1_list)),
                   "test_acc_mean": float(np.mean(acc_list)),
                   "binary_weights": rows[0]["binary_weights"], "per_seed": rows}}
    json.dump(res, open(f"{OUT}/bnn_baseline_{a.input}.json", "w"), indent=2)
    log(f"[done] bnn-{a.input} TEST f1={np.mean(f1_list):.4f}"
        f"+-{np.std(f1_list):.4f} acc={np.mean(acc_list):.4f}")


if __name__ == "__main__":
    main()
