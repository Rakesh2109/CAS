"""Recover full-pool positive/negative supports for Table III, seed 42."""
from pathlib import Path
from multiprocessing import Pool
import json,time,warnings
import numpy as np
import heldout_protocol as hp
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',message='Inconsistent values: penalty')
hp.BMAX=10;hp.B_SNAPS=(10,)
OUT=Path(hp.OUT);AUDIT=OUT/'full_signed_seed42';AUDIT.mkdir(exist_ok=True)
def job(fi):
 t=time.time();y=np.load(Path(hp.BASE)/hp.NPZ)['ytr'].astype(int)
 _,cp,_=hp.train_split(y)
 Z=np.asarray(np.load(OUT/'Z_cpss_seed42.npy',mmap_mode='r'),np.float32)
 pi,cf=hp.cpss_snap(Z,(y[cp]==fi).astype(int),42+fi)[10]
 d=dict(pi=pi,coef=cf)
 (AUDIT/f'class{fi}.json').write_text(json.dumps(d))
 print(f'Done {hp.FAMILIES[fi]} in {time.time()-t:.1f}s',flush=True)
 return d
if __name__=='__main__':
 with Pool(6) as p:rows=p.map(job,range(6))
 pos=[];neg=[]
 for d in rows:
  pi=np.array(d['pi']);cf=np.array(d['coef']);pos.append(int(((pi>=.9)&(cf>0)).sum()));neg.append(int(((pi>=.9)&(cf<0)).sum()))
 saved=json.load(open(OUT/'heldout_cas.json'))['pools']['full']['cas']['per_seed'][0]
 assert pos==saved['S_pos'],(pos,saved['S_pos'])
 Z=np.asarray(np.load(OUT/'Z_test_seed42.npy'),np.float32);yt=np.load(Path(hp.BASE)/hp.NPZ)['yte'].astype(int)
 f,a,_=hp.score(Z,yt,[np.array(d['pi']) for d in rows],[np.array(d['coef']) for d in rows],.9)
 assert abs(f-saved['test_f1'])<1e-12 and abs(a-saved['test_acc'])<1e-12
 summary=dict(seed=42,B=10,threshold=.9,positive=pos,negative=neg,positive_sum=sum(pos),negative_sum=sum(neg),scores_match=True)
 (AUDIT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
