import sys,json,time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0,'/FPTM/CAS_IoMT_Empirical/scripts')
import heldout_protocol as hp
ROOT=Path('/FPTM/CAS_IoMT_Empirical')
OUT=ROOT/'results/heldout'
AUDIT=ROOT/'paper/review/signed_support_audit'
AUDIT.mkdir(exist_ok=True)
hp.BMAX=10;hp.B_SNAPS=(5,10)
def job(args):
 seed,fi=args
 path=AUDIT/f'pruned_seed{seed}_class{fi}.json'
 if path.exists():return seed,fi,json.loads(path.read_text())
 t=time.time()
 y=np.load(ROOT/'results/booleanized_data.npz')['ytr'].astype(int)
 _,cp,_=hp.train_split(y)
 cols=hp.keep_cols(np.load(OUT/f'W_seed{seed}.npy'),60)
 Z=np.asarray(np.load(OUT/f'Z_cpss_seed{seed}.npy',mmap_mode='r')[:,cols],np.float32)
 snaps=hp.cpss_snap(Z,(y[cp]==fi).astype(int),seed+fi)
 result={str(k):{'pi':v[0],'coef':v[1]} for k,v in snaps.items()}
 path.write_text(json.dumps(result))
 print(f'Completed seed {seed}, class {hp.FAMILIES[fi]} in {time.time()-t:.1f}s',flush=True)
 return seed,fi,result
if __name__=='__main__':
 with Pool(6) as pool: rows=pool.map(job,[(s,f) for s in hp.SEEDS for f in range(6)])
 snapshots={(s,f):d for s,f,d in rows}
 saved=json.load(open(OUT/'heldout_cas.json'))['pools']['pruned']['grid']
 summary={}
 for B,thr in [(5,.8),(10,.9),(10,.5)]:
  key=f'{B}|{thr}'; per=[]
  for seed in hp.SEEDS:
   pos=[];neg=[];unique=set()
   for f in range(6):
    d=snapshots[seed,f][str(B)];pi=np.array(d['pi']);cf=np.array(d['coef'])
    pp=np.flatnonzero((pi>=thr)&(cf>0));nn=np.flatnonzero((pi>=thr)&(cf<0))
    pos.append(len(pp));neg.append(len(nn));unique.update(pp.tolist());unique.update(nn.tolist())
   expected=next(r['S_pos'] for r in saved[key]['per_seed'] if r['seed']==seed)
   assert pos==expected,(seed,key,pos,expected)
   per.append(dict(seed=seed,positive=pos,negative=neg,positive_sum=sum(pos),negative_sum=sum(neg),signed_sum=sum(pos)+sum(neg),distinct_signed=len(unique)))
  summary[key]={'per_seed':per,'means':{k:float(np.mean([r[k] for r in per])) for k in ['positive_sum','negative_sum','signed_sum','distinct_signed']}}
 (AUDIT/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
