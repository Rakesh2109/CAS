"""Diagrams for CPSS_explained_simple.tex. Real data where real data exists."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = "/FPTM/CAS_IoMT_Empirical/figs_explain"
FAM = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
BLUE, ORANGE, GREY, GREEN, RED = "#2f5d8a", "#c8752a", "#9aa3ad", "#2f8f4e", "#b32d2d"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def load():
    Z = np.asarray(np.load("/FPTM/CAS_IoMT_Empirical/results/Zte_seed42.npy", mmap_mode="r")[:20000])
    y = np.load("/FPTM/CAS_IoMT_Empirical/results/booleanized_data.npz")["yte"][:20000]
    return Z, y


# ---------------------------------------------------------------- fig 1: what Z is
def fig_z_anatomy(Z, y):
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.28)

    # left: schematic
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.6, "One row of $Z$ = one network flow", ha="center", fontsize=8.5, weight="bold")
    rng = np.random.RandomState(3)
    cellw, cellh = 0.62, 0.62
    x0, y0 = 1.5, 3.15
    for r in range(8):
        for c in range(12):
            v = rng.rand() < (0.30 if 3 <= c <= 5 else 0.12)
            ax.add_patch(Rectangle((x0 + c * cellw, y0 + (7 - r) * cellh), cellw, cellh,
                                   fc=BLUE if v else "white", ec="#cfd4da", lw=0.4))
            ax.text(x0 + c * cellw + cellw / 2, y0 + (7 - r) * cellh + cellh / 2,
                    "1" if v else "0", ha="center", va="center", fontsize=5.2,
                    color="white" if v else "#b0b6bd")
    ax.text(x0 - 0.25, y0 + 7 * cellh + cellh / 2, "flow 1", ha="right", va="center", fontsize=7)
    ax.text(x0 - 0.25, y0 + 6 * cellh + cellh / 2, "flow 2", ha="right", va="center", fontsize=7)
    ax.text(x0 - 0.25, y0 + 0 * cellh + cellh / 2, "flow $n$", ha="right", va="center", fontsize=7)
    ax.text(x0 + 6 * cellw, y0 - 0.55, "2,400 clause columns  $\\rightarrow$", ha="center", fontsize=7.5)
    ax.annotate("", xy=(x0 + 3.5 * cellw, y0 + 8 * cellh + 0.15), xytext=(x0 + 3.5 * cellw, y0 + 8 * cellh + 1.0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    ax.text(x0 + 3.5 * cellw, y0 + 8 * cellh + 1.15, "one clause = one IF-THEN rule",
            ha="center", fontsize=7, color=ORANGE)
    ax.text(5, 1.45, "$z_j(x)=1$  the rule matched this flow\n$z_j(x)=0$  it did not",
            ha="center", fontsize=7.5)
    ax.text(5, 0.30, "no raw feature values live in $Z$ - only rule hits",
            ha="center", fontsize=7.5, style="italic", color=RED)

    # right: real activations
    ax2 = fig.add_subplot(gs[1])
    idx = np.concatenate([np.where(y == c)[0][:10] for c in range(6)])
    ax2.imshow(Z[idx], aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    for k in range(1, 6):
        ax2.axvline(k * 400 - 0.5, color=ORANGE, lw=0.9)
    for k in range(1, 6):
        ax2.axhline(k * 10 - 0.5, color="#666", lw=0.5, ls=":")
    ax2.set_xticks([200 + 400 * k for k in range(6)])
    ax2.set_xticklabels(FAM, fontsize=6.5, rotation=30, ha="right")
    ax2.set_yticks([5 + 10 * k for k in range(6)])
    ax2.set_yticklabels(FAM, fontsize=6.5)
    ax2.set_xlabel("clause banks (400 clauses each)", fontsize=7.5)
    ax2.set_ylabel("10 real test flows per class", fontsize=7.5)
    ax2.set_title("Real $Z$ from our run (dark = clause fired)", fontsize=8.5, weight="bold")
    ax2.tick_params(length=2)
    fig.savefig(f"{OUT}/fig_z_anatomy.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: complementary pairs
def fig_pairs():
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")
    rng = np.random.RandomState(1)
    n = 12
    ax.text(1.4, 5.7, "fit rows", ha="center", fontsize=8, weight="bold")
    for i in range(n):
        ax.add_patch(Rectangle((1.0, 5.2 - i * 0.34), 0.8, 0.3, fc="#dde4ec", ec="#a9b4c0", lw=0.4))
        ax.text(1.4, 5.35 - i * 0.34, f"{i+1}", ha="center", va="center", fontsize=5.5)
    for b, xb in enumerate([4.2, 7.6, 11.0]):
        perm = rng.permutation(n)
        h1, h2 = perm[:6], perm[6:]
        ax.text(xb + 0.9, 5.9, f"pair $b={b+1}$" if b < 2 else "...", ha="center", fontsize=8, weight="bold")
        for hi, (h, col, lab) in enumerate([(h1, BLUE, "half 1"), (h2, ORANGE, "half 2")]):
            xx = xb + hi * 1.8
            ax.text(xx + 0.4, 5.45, lab, ha="center", fontsize=6.5, color=col)
            for i, r in enumerate(sorted(h)):
                ax.add_patch(Rectangle((xx, 5.0 - i * 0.34), 0.8, 0.3, fc=col, ec="none", alpha=0.75))
                ax.text(xx + 0.4, 5.15 - i * 0.34, f"{r+1}", ha="center", va="center",
                        fontsize=5.5, color="white")
        ax.add_patch(FancyArrowPatch((1.95, 3.4), (xb - 0.15, 3.4), arrowstyle="-|>",
                                     mutation_scale=8, color="#8b939c", lw=0.7,
                                     connectionstyle=f"arc3,rad={-0.18 - 0.06*b}"))
        if b == 0:
            ax.text((1.95 + xb) / 2, 2.75, "shuffle + cut", fontsize=6.2,
                    color="#8b939c", ha="center")
    ax.text(7.0, 0.75, "the two halves of a pair are DISJOINT and together cover the data exactly once"
                       "  $\\Rightarrow$  'complementary'", ha="center", fontsize=7.5, color=GREEN)
    ax.text(7.0, 0.15, "$B$ pairs  $\\Rightarrow$  $2B$ halves  ($B{=}15 \\Rightarrow 30$ fits per class)",
            ha="center", fontsize=8, weight="bold")
    fig.savefig(f"{OUT}/fig_pairs.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: vote pipeline
def fig_vote():
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    rng = np.random.RandomState(7)
    truth = [1.0, 0.9, 0.5, 0.3]
    names = ["$c_1$", "$c_2$", "$c_6$", "$c_4$"]
    nh, ng = 8, 6
    ax.text(3.9, 7.5, "step 1: on each half, fit $\\ell_1$ logistic at 6 values of $\\kappa$",
            ha="center", fontsize=8, weight="bold")
    ax.text(3.9, 7.0, "cell dark = that fit gave the clause a non-zero coefficient", ha="center", fontsize=6.8)
    sel = np.zeros((4, nh), bool)
    for ci, p in enumerate(truth):
        yb = 6.1 - ci * 1.45
        ax.text(0.35, yb - 0.35, names[ci], fontsize=8, ha="center")
        for h in range(nh):
            hit = rng.rand() < p
            sel[ci, h] = hit
            for g in range(ng):
                on = hit and (rng.rand() < 0.55 or g >= 3)
                ax.add_patch(Rectangle((0.9 + h * 0.88 + g * 0.13, yb - 0.62), 0.12, 0.55,
                                       fc=BLUE if on else "#e9edf1", ec="none"))
            ax.add_patch(Rectangle((0.87 + h * 0.88, yb - 0.65), 0.81, 0.61, fc="none",
                                   ec=GREEN if hit else "#c9ced4", lw=0.9))
        ax.text(0.9 + nh * 0.88 + 0.15, yb - 0.35, "$\\rightarrow$", fontsize=9, va="center")
    ax.text(4.3, 0.72, "half 1        half 2        ...        half $2B$", fontsize=6.8, ha="center")
    ax.text(3.9, 0.18, "green box = selected on that half (UNION over the 6 $\\kappa$)",
            ha="center", fontsize=7, color=GREEN)

    # right panel: pi_hat bars
    x0 = 9.4
    ax.text(12.0, 7.5, "step 2: count votes, threshold, read sign", ha="center", fontsize=8, weight="bold")
    thr = 0.8
    pis = [1.0, 0.9, 0.5, 0.3]
    coefs = ["$+0.9$", "$-0.4$", "$+0.3$", "$+0.1$"]
    for ci in range(4):
        yb = 6.1 - ci * 1.45
        w = pis[ci] * 3.6
        keep = pis[ci] >= thr
        ax.add_patch(Rectangle((x0, yb - 0.58), 3.6, 0.48, fc="#eef1f4", ec="#c9ced4", lw=0.5))
        ax.add_patch(Rectangle((x0, yb - 0.58), w, 0.48, fc=GREEN if keep else RED, alpha=0.75, ec="none"))
        ax.text(x0 + 3.75, yb - 0.34, f"$\\hat\\pi={pis[ci]:.2f}$", fontsize=7, va="center")
        ax.text(x0 + 5.05, yb - 0.34, coefs[ci], fontsize=7, va="center",
                color=BLUE if "+" in coefs[ci] else ORANGE)
        tag = ("$S^+$" if "+" in coefs[ci] else "$S^-$") if keep else "drop"
        ax.text(x0 + 5.9, yb - 0.34, tag, fontsize=7.5, va="center", weight="bold",
                color=GREEN if keep else RED)
    ax.plot([x0 + thr * 3.6] * 2, [1.05, 6.25], color=RED, lw=1.1, ls="--")
    ax.text(x0 + thr * 3.6, 6.45, "$\\pi_{\\mathrm{thr}}=0.8$", fontsize=7, color=RED, ha="center")
    ax.text(x0 + 1.8, 0.30, "kept if selected in $\\geq$ 80% of halves;\nsign of mean coef splits $S^+$ / $S^-$",
            ha="center", fontsize=7)
    fig.savefig(f"{OUT}/fig_vote.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: literal collapse
def fig_collapse():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(2.1, 5.6, "raw literals in the clause", ha="center", fontsize=8, weight="bold")
    lits = ["Header_Length>=bin2", "Header_Length>=bin3", "NOT(Header_Length>=bin4)",
            "NOT(Header_Length>=bin5)", "NOT(Header_Length>=bin6)", "... up to bin9"]
    for i, l in enumerate(lits):
        ax.add_patch(Rectangle((0.15, 4.9 - i * 0.58), 3.9, 0.46,
                               fc="#eef1f4" if i > 1 else "#e3ecf5", ec="#c9ced4", lw=0.5))
        ax.text(0.32, 5.13 - i * 0.58, l, fontsize=6.4, va="center", family="monospace")
    ax.annotate("", xy=(5.55, 3.4), xytext=(4.25, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    ax.text(4.9, 3.62, "collapse", fontsize=7, color=ORANGE, ha="center")

    # thermometer axis
    xs, xe = 5.9, 11.6
    bins = 10
    w = (xe - xs) / bins
    for k in range(bins):
        inside = k == 2
        ax.add_patch(Rectangle((xs + k * w, 3.65), w, 0.5,
                               fc=GREEN if inside else "#eef1f4", ec="#b9c0c8", lw=0.5, alpha=0.85))
        ax.text(xs + k * w + w / 2, 3.40, f"b{k+1}", fontsize=5.6, ha="center", color="#6b7480")
    ax.text(xs + 2.5 * w, 4.35, "the only band left", fontsize=6.8, ha="center", color=GREEN)
    ax.annotate("", xy=(xs + 2 * w, 4.25), xytext=(xs + 2.5 * w, 4.30),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    ax.text((xs + xe) / 2, 2.75,
            "$\\mathtt{bin3} \\leq \\mathtt{Header\\_Length} < \\mathtt{bin4}$",
            fontsize=9, ha="center")
    ax.text((xs + xe) / 2, 1.95, "many '$\\geq$' literals $\\rightarrow$ keep the strongest lower bound\n"
                                "many 'NOT $\\geq$' literals $\\rightarrow$ keep the weakest upper bound",
            fontsize=6.8, ha="center", color="#5a6570")
    ax.add_patch(Rectangle((0.15, 0.20), 11.7, 0.85, fc="#f3f4f6", ec="#d6dade", lw=0.6))
    ax.text(6.0, 0.62, "real clause #1737 in our DDoS signature:  130 literals  $\\longrightarrow$  "
                       "22 feature conditions  $\\longrightarrow$  4 informative ones",
            fontsize=8, ha="center", weight="bold")
    fig.savefig(f"{OUT}/fig_collapse.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: real bank structure
def fig_banks(Z, y):
    pos = np.zeros((6, 6)); neg = np.zeros((6, 6))
    for c in range(6):
        m = Z[y == c]
        for k in range(6):
            pos[c, k] = m[:, k * 400:k * 400 + 200].sum(1).mean()
            neg[c, k] = m[:, k * 400 + 200:(k + 1) * 400].sum(1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, M, t in [(axes[0], pos, "positive-polarity clauses\n(vote FOR their own class)"),
                     (axes[1], neg, "negative-polarity clauses\n(vote AGAINST their own class)")]:
        im = ax.imshow(M, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6.2,
                        color="white" if M[i, j] > M.max() * 0.55 else "#33383d")
        ax.set_xticks(range(6)); ax.set_xticklabels(FAM, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks(range(6)); ax.set_yticklabels(FAM, fontsize=6.5)
        ax.set_xlabel("clause bank", fontsize=7.5)
        ax.set_title(t, fontsize=8, weight="bold")
        ax.tick_params(length=2)
        for i in range(6):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fc="none", ec=RED, lw=1.3))
    axes[0].set_ylabel("true class of the flow", fontsize=7.5)
    fig.savefig(f"{OUT}/fig_banks.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    Z, y = load()
    fig_z_anatomy(Z, y)
    fig_pairs()
    fig_vote()
    fig_collapse()
    fig_banks(Z, y)
    print("figures written to", OUT)
