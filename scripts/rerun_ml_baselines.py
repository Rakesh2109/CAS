"""Rerun numeric- and binary-input ML baselines on fixed CAS partitions.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/rerun_ml_baselines.py.
Outputs are separate from historical results; no SMOTE or TM retraining.
"""
import os,json,time,hashlib,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from joblib import parallel_backend
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import ConvergenceWarning
import sklearn
import heldout_protocol as hp
import nontm_baselines as nb
from raw_feature_baselines import raw_capped
ROOT=Path('/FPTM/CAS_IoMT_Empirical')
OUT=ROOT/'results/heldout/ml_rerun_20260911'
nb.N_JOBS=4

def run_job(args):
 rep,name,seed=args
 cache=OUT/f'{rep}_{name}_seed{seed}.json'
 if cache.exists():return rep,name,json.loads(cache.read_text())
 d=np.load(OUT/f'{rep}_data.npz');X=d['Xtr'];y=d['ytr'];Xt=d['Xte'];yt=d['yte']
 core,cp,sel=hp.train_split(y);fit=np.sort(np.r_[core,cp]);Xc=X[fit];Xs=X[sel];yc=y[fit];ys=y[sel]
 scaler=None
 if rep=='numeric' and name=='l1':
  scaler=StandardScaler().fit(Xc);Xc=scaler.transform(Xc);Xs=scaler.transform(Xs);Xt=scaler.transform(Xt)
 fn,grid=nb.GRIDS[name];runs=[]
 if rep=="binary" and name=="xgb":grid=[(4,200)]  # repeat historical selected configuration
 best=-1.;bm=None;trials=[]
 for param in grid:
  t=time.time()
  if name=='l1':
   limit=2000 if rep=='numeric' else 200
   with warnings.catch_warnings(record=True) as ws, parallel_backend('loky', inner_max_num_threads=1):
    warnings.simplefilter('always',ConvergenceWarning)
    m=OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear',C=param,max_iter=limit,class_weight='balanced',random_state=seed),n_jobs=6).fit(Xc,yc)
   conv=any(isinstance(w.message,ConvergenceWarning) for w in ws)
   if conv: raise RuntimeError(f'Convergence failure: {rep} {name} {seed} {param}')
  else:m=fn(Xc,yc,seed,param)
  sf=float(hp.f1_score(ys,m.predict(Xs),average='macro'))
  trials.append(dict(hp=param,selection_macro_f1=sf,seconds=time.time()-t))
  print(f'{rep} {name} seed={seed} hp={param} selection={sf:.6f} ({time.time()-t:.1f}s)',flush=True)
  if sf>best:best=sf;bm=m;best_hp=param
 pred=bm.predict(Xt)
 row=dict(seed=seed,hp=best_hp,selection_macro_f1=best,test_f1=float(hp.f1_score(yt,pred,average='macro')),test_acc=float(np.mean(pred==yt)),prediction_sha256=hashlib.sha256(pred.astype(np.int64).tobytes()).hexdigest(),trials=trials)
 if name=='l1':row['nonzero_coefficients']=int(sum(np.count_nonzero(np.abs(e.coef_)>1e-8) for e in bm.estimators_));row['iterations']=[int(e.n_iter_[0]) for e in bm.estimators_]
 np.save(OUT/f'{rep}_{name}_seed{seed}_predictions.npy',pred)
 runs.append(row)
 print(f'DONE {rep} {name} seed={seed}: {row["test_f1"]:.8f}',flush=True)
 result=dict(representation=rep,model=name,n_features=X.shape[1],training_rows=len(fit),selection_rows=len(sel),test_rows=len(yt),seeds=hp.SEEDS,split_seed=0,smote=False,l1_standardized=bool(scaler is not None),sklearn_version=sklearn.__version__,per_seed=runs)
 for metric in ['test_f1','test_acc']:
  result[metric+'_mean']=float(np.mean([r[metric] for r in runs]));result[metric+'_sd']=float(np.std([r[metric] for r in runs]))
 cache.write_text(json.dumps(result,indent=2))
 return rep,name,result

if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 if not (OUT/'numeric_data.npz').exists():
  X,y,Xt,yt=hp.load()
  (R,ry),(Rt,ryt)=raw_capped()
  assert np.array_equal(y,ry) and np.array_equal(yt,ryt)
  assert np.isfinite(R).all() and np.isfinite(Rt).all()
  assert R.shape==(195338,22) and Rt.shape==(80868,22)
  for rep,a,b in [('binary',X,Xt),('numeric',R,Rt)]:np.savez(OUT/f'{rep}_data.npz',Xtr=a.astype(np.float32),ytr=y,Xte=b.astype(np.float32),yte=yt)
 complete={}
 jobs=[]
 for rep in ['binary','numeric']:
  for name in ['dt','l1','rf','xgb']:
   f=OUT/f'{rep}_{name}.json'
   if f.exists():complete[rep+'_'+name]=json.loads(f.read_text())
   else:jobs.extend((rep,name,seed) for seed in hp.SEEDS)
 with ProcessPoolExecutor(max_workers=9) as pool:results=list(pool.map(run_job,jobs))
 for rep,name,result in results:
  key=rep+'_'+name
  if key not in complete:complete[key]={**result,'per_seed':[]}
  complete[key]['per_seed']+=result['per_seed']
 for key,result in complete.items():
  result['per_seed'].sort(key=lambda r:hp.SEEDS.index(r['seed']))
  assert len(result['per_seed'])==3
  for metric in ['test_f1','test_acc']:
   result[metric+'_mean']=float(np.mean([r[metric] for r in result['per_seed']]))
   result[metric+'_sd']=float(np.std([r[metric] for r in result['per_seed']]))
  (OUT/f'{key}.json').write_text(json.dumps(result,indent=2))
 (OUT/'summary.json').write_text(json.dumps(complete,indent=2))
 print('All baseline reruns complete.',flush=True)
