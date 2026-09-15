"""Build Figure 1 as editable SVG and publication-size vector PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(7.16, 1.70))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 7.16), ylim=(0, 1.70))
ax.axis("off")
INK, MUTED = "#243247", "#566477"
BLUE, PURPLE, GREEN = "#2457a7", "#68439b", "#23754d"


def text(x, y, label, size=8, color=INK, weight="normal", ha="center"):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", linespacing=1.15)


def box(x, y, w, h, label, color=BLUE, fill="#f0f5fc", size=8):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.055",
                 linewidth=1, edgecolor=color, facecolor=fill, zorder=2))
    text(x, y, label, size=size)


def arrow(points, color=INK, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                linewidth=1, linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=9, linewidth=1, color=color,
                 linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0,
                 zorder=1))


# Two aligned rows: extraction above, inference below.
text(.12, 1.58, "SIGNATURE EXTRACTION", 8, BLUE, "bold", "left")
box(.96, 1.19, 1.65, .50, "Training partition\nBinarizer + weighted TM", size=8)
box(2.82, 1.19, 1.62, .50,
    "Optional pool pruning\nRemove smallest $|w_{c,j}|$", size=8)
box(4.62, 1.19, 1.60, .50,
    "CPSS on fit partition\nComplementary halves\nUnion over regularization grid",
    PURPLE, "#f5f1fa", 7.5)
box(6.34, 1.19, 1.40, .50,
    "Stable signed supports\nThreshold stability votes\nSplit by coefficient sign",
    GREEN, "#f0f8f3", 7.5)
arrow([(1.80,1.19),(1.99,1.19)])
arrow([(3.65,1.19),(3.80,1.19)])
arrow([(5.44,1.19),(5.62,1.19)])
# Keep selection text below its connector, so neither branch crosses text.
text(.12,.72,"Selection partition: choose pruning budget and CPSS settings; then freeze.",
     7.5,PURPLE,ha="left")
ax.plot([2.82,4.62],[.845,.845],color=PURPLE,linewidth=1,
        linestyle="--",zorder=1)
arrow([(2.82,.845),(2.82,.925)],PURPLE,True)
arrow([(4.62,.845),(4.62,.925)],PURPLE,True)
text(.12,.60,"SCORING",8,BLUE,"bold","left")
box(.96,.30,1.65,.43,"Report partition\nHeld-out flow $x$",size=8)
box(2.82,.30,1.62,.43,"Same retained pool\nActivations $z(x)$",size=8)
box(4.62,.30,1.60,.43,"Normalized signed score\n$s_\\ell(x)$",size=8)
box(6.34,.30,1.40,.43,"Predicted class\n$\\hat y$",size=8)
arrow([(1.80,.30),(1.99,.30)])
arrow([(3.65,.30),(3.80,.30)])
arrow([(5.44,.30),(5.62,.30)])
arrow([(6.34,.925),(6.34,.64),(4.62,.64),(4.62,.535)],GREEN)

for extension in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_cpss_protocol.{extension}", dpi=240,
                facecolor="white")
plt.close(fig)
