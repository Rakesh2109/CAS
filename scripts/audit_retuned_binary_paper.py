"""Verify subsection D from cached tuned activations, weights and CPSS snapshots."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from binary_retune_rerun import score_matrix, splits
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results'
TAG='_c1000t1000'
exp=json.loads((BASE/f'binary_retuned_pruned{TAG}.json').read_text())
d=np.load(BASE/'booleanized_data.npz'); y=(d['yte']!=0).astype(int)
_,sel,rep=splits(len(y)); nc=exp['config']['clauses']
checks=[]
def close(a,b,label,tol=1e-12):
    assert abs(a-b)<=tol,(label,a,b)
    checks.append(label)
def stats(vals):return {'mean':float(np.mean(vals)), 'sd':float(np.std(vals))}
def metrics(Y,S):
    pred=S.argmax(1)
    return {'macro_f1':float(f1_score(Y,pred,average='macro')),
            'accuracy':float((Y==pred).mean()),'attack_f1':float(f1_score(Y,pred)),
            'benign_f1':float(f1_score(Y,pred,pos_label=0)),
            'attack_precision':float(precision_score(Y,pred)),
            'attack_recall':float(recall_score(Y,pred)),
            'benign_recall':float(recall_score(Y,pred,pos_label=0)),
            'auroc_margin':float(roc_auc_score(Y,S[:,1]-S[:,0])),
            'auroc_raw_attack':float(roc_auc_score(Y,S[:,1]))}
out={'source':f'results/binary_retuned_pruned{TAG}.json','sd_ddof':0,'pools':{}}
sm=np.array(exp['prune_curve_sel']).mean(0)
close(exp['ks'][np.flatnonzero(sm>=sm[0]-.01)[-1]],exp['kstar_removed_per_class'],'selection pruning rule')
for pool,info in exp['pools'].items():
    best=max(info['grid'],key=lambda k:info['grid'][k]['sel_f1'])
    assert best==f"{info['selected_B']}|{info['selected_pi_thr']}"
    base_rows=[]; cas_rows=[]
    for si,seed in enumerate([42,7,123]):
        Z=np.load(BASE/f'Zte_binary_retuned{TAG}_seed{seed}.npy',mmap_mode='r')
        W=np.load(BASE/f'W_binary_retuned{TAG}_seed{seed}.npy')
        keep=info['clauses_per_class']
        cols=np.concatenate([c*nc+np.sort(np.argsort(abs(W[c*nc:(c+1)*nc]),kind='stable')[nc-keep:]) for c in range(2)])
        zr=np.asarray(Z[rep][:,cols],np.float32); zs=np.asarray(Z[sel][:,cols],np.float32);wk=W[cols]
        tm=np.column_stack([zr[:,c*keep:(c+1)*keep]@wk[c*keep:(c+1)*keep] for c in range(2)])
        bm=metrics(y[rep],tm)
        bm['auroc_clipped_attack']=float(roc_auc_score(y[rep],np.clip(tm[:,1],0,exp['config']['T'])/exp['config']['T']))
        base_rows.append(bm)
        snaps=[json.loads((BASE/f'binary_retuned_snaps{TAG}_{pool}_seed{seed}_class{c}.json').read_text()) for c in range(2)]
        for key,g in info['grid'].items():
            b,t=key.split('|');t=float(t)
            pis=[np.array(s[b][0]) for s in snaps];cfs=[np.array(s[b][1]) for s in snaps]
            expected=next(r for r in g['per_seed'] if r['seed']==seed)
            ss=score_matrix(zs,pis,cfs,t);sr=score_matrix(zr,pis,cfs,t)
            close(f1_score(y[sel],ss.argmax(1),average='macro'),expected['sel_f1'],f'{pool}/{seed}/{key}/selection F1')
            close(f1_score(y[rep],sr.argmax(1),average='macro'),expected['rep_f1'],f'{pool}/{seed}/{key}/report F1')
            close((sr.argmax(1)==y[rep]).mean(),expected['rep_acc'],f'{pool}/{seed}/{key}/accuracy')
            close(roc_auc_score(y[rep],sr[:,1]-sr[:,0]),expected['rep_auroc'],f'{pool}/{seed}/{key}/AUROC',2e-5)
            for c in range(2):
                close(int((pis[c]>=t).sum()),expected['S'][c],f'{pool}/{seed}/{key}/{c}/support')
                close(int(((pis[c]>=t)&(cfs[c]>0)).sum()),expected['S_pos'][c],f'{pool}/{seed}/{key}/{c}/positive support')
            if key==best:cas_rows.append(metrics(y[rep],sr))
    summary={name:{k:stats([r[k] for r in rows]) for k in rows[0]} for name,rows in [('tm',base_rows),('cas',cas_rows)]}
    for metric,saved in [('macro_f1','base_rep_f1'),('accuracy','base_rep_acc'),('auroc_margin','base_rep_auroc')]:
        close(summary['tm'][metric]['mean'],info[saved],f'{pool}/baseline {metric}')
        close(summary['tm'][metric]['sd'],info[saved+'_sd'],f'{pool}/baseline {metric} SD')
    summary['selected']=best
    summary['matched_signature_sizes']=info['grid']['15|0.8']['per_seed'][0]
    summary['sweeps']={}
    for name,keys in [('threshold',[f'15|{v}' for v in [.5,.6,.7,.8,.9]]),('pairs',[f'{v}|0.8' for v in [5,10,15,20,25]])]:
        vals=[info['grid'][k]['rep_f1'] for k in keys]
        summary['sweeps'][name]={'min':min(vals),'max':max(vals),'span':max(vals)-min(vals)}
    out['pools'][pool]=summary
    print(pool,summary,flush=True)
grid1=json.loads((BASE/'binary_tm_gridsearch.json').read_text());grid2=json.loads((BASE/'binary_tm_gridsearch2.json').read_text())
winner=max(grid2['grid'],key=lambda r:r['sel_mean'])
assert all(winner[k]==exp['config'][k] for k in ['clauses','T','s','balance'])
assert winner['epoch_at_mean']==exp['config']['epochs']
out['grid_search']={'stage1_configs':len(grid1['grid']),'stage2_configs':len(grid2['grid']),'stage2_winner':winner,'seed':grid2['seed']}
out['checks_passed']=len(checks)
(ROOT/'paper/review/black_results_review/results_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('Passed',len(checks),'numeric checks.',flush=True)
