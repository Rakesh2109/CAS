"""Binary counterpart of Figure 3, with selection-chosen pruning budget."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threeway_binary_pruned import prep, splits, class_sums, SEEDS, NC, NCLS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results'
exp = json.loads((BASE / 'threeway_binary_pruned.json').read_text())
y = prep()
_, _, rep = splits(len(y))
ks = np.array(exp['ks'])
f1, acc, class_f1 = [], [], []
for seed in SEEDS:
    Z = np.asarray(np.load(BASE / f'Zte_binary_seed{seed}.npy', mmap_mode='r')[rep], dtype=np.float32)
    W = np.load(BASE / f'W_binary_seed{seed}.npy')
    orders = [c * NC + np.argsort(np.abs(W[c*NC:(c+1)*NC]), kind='stable') for c in range(NCLS)]
    ff, aa, cc = [], [], []
    for k in ks:
        wm = W.copy()
        if k:
            wm[np.concatenate([order[:k] for order in orders])] = 0
        pred = class_sums(Z, wm, NC).argmax(1)
        class_scores = f1_score(y[rep], pred, labels=[0, 1], average=None, zero_division=0)
        cc.append(class_scores.tolist())
        ff.append(float(class_scores.mean()))
        aa.append(float((y[rep] == pred).mean()))
    f1.append(ff)
    acc.append(aa)
    class_f1.append(cc)
f1, acc, class_f1 = np.array(f1), np.array(acc), np.array(class_f1)
assert np.allclose(f1, class_f1.mean(axis=2), atol=1e-12, rtol=0)
assert np.allclose(f1, exp['prune_curve_rep'], atol=1e-12, rtol=0)
kstar = exp['kstar_removed_per_class']
selection_mean = np.array(exp['prune_curve_sel']).mean(0)
assert kstar == ks[np.where(selection_mean >= selection_mean[0] - .01)[0][-1]]
audit = json.loads((ROOT / 'paper/review/binary_revision/results_audit.json').read_text())
ki = list(ks).index(kstar)
assert abs(acc[:,ki].mean()-audit['pools']['pruned']['tm']['accuracy']['mean']) < 1e-12
result = {'seeds': SEEDS, 'ks': ks.tolist(), 'selected_k': kstar,
          'budget_selection_partition': 'selection quarter', 'plot_partition': 'report quarter',
          'macro_f1_per_seed': f1.tolist(), 'accuracy_per_seed': acc.tolist(),
          'class_order': ['Benign', 'Attack'], 'class_f1_per_seed': class_f1.tolist()}
(ROOT / 'figs/fig_binary_clause_pruning.json').write_text(json.dumps(result, indent=2)+'\n')
plt.rcParams.update({'font.size': 12, 'text.color': 'blue', 'axes.labelcolor': 'blue',
                     'xtick.color': 'blue', 'ytick.color': 'blue', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(5.0, 3.3))
for vals, color, label, style in [
        (f1, 'C1', 'macro-F1', '-'), (acc, 'C0', 'accuracy', '-'),
        (class_f1[:, :, 0], 'C2', 'Benign F1', '--'),
        (class_f1[:, :, 1], 'C3', 'Attack F1', '-.')]:
    mean, sd = vals.mean(0), vals.std(0)
    ax.fill_between(ks,mean-sd,mean+sd,color=color,alpha=.18)
    ax.plot(ks,mean,color=color,lw=1.6,ls=style,label=label)
ax.axhline(f1.mean(0)[0],ls='--',lw=1,color='.45',label='full-pool macro-F1')
ax.axvline(kstar,ls=':',lw=1.2,color='.25')
ax.annotate('$k^*=385$ removed/class\n30 clauses retained',
            xy=(kstar,f1.mean(0)[ki]),xytext=(100,.47),fontsize=10,
            arrowprops={'arrowstyle':'->','lw':.8,'color':'.25'})
ax.set_xlabel('clauses removed per class')
ax.set_ylabel('report-quarter score')
ax.set_xlim(0,NC)
ax.set_ylim(0,1)
ax.grid(ls='--',alpha=.3)
ax.legend(fontsize=9,loc='lower left',ncol=2,columnspacing=1,handlelength=2)
fig.tight_layout()
for ext in ('pdf','png','svg'):
    fig.savefig(ROOT / f'figs/fig_binary_clause_pruning.{ext}',dpi=220)
plt.close(fig)
for k in [0,385,390,395]:
    i=list(ks).index(k)
    print(k,'macro-F1',f1.mean(0)[i],'+/-',f1.std(0)[i],'accuracy',acc.mean(0)[i],'+/-',acc.std(0)[i])

for k in [0, kstar]:
    i = list(ks).index(k)
    print(k, 'class F1 means', class_f1[:, i, :].mean(0), 'SD', class_f1[:, i, :].std(0))
