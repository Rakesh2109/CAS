import json,pickle,numpy as np,sys
sys.path.insert(0,'scripts')
from full_cas import BASE,FAM_MC,stratified_subsample,cpss_full,cas_scores,CAP_PER_CLASS,SUBSEED
from sklearn.metrics import classification_report
d=np.load(f'{BASE}/dedup_bool_nbins10.npz'); Xte,yte=d['Xte'],d['yte']
sub=stratified_subsample(yte,CAP_PER_CLASS,SUBSEED); ysub=yte[sub]
rng=np.random.RandomState(0); perm=rng.permutation(len(sub))
val,ev=perm[:len(sub)//2],perm[len(sub)//2:]
tm=pickle.load(open(f'{BASE}/tm_dedup_multiclass_seed42.pkl','rb'))
Z=tm.transform(np.ascontiguousarray(Xte[sub],np.uint32)).astype(np.float32)
bp=tm.predict(np.ascontiguousarray(Xte[sub][ev],np.uint32))
print('BASE:'); print(classification_report(ysub[ev],bp,target_names=FAM_MC,zero_division=0,digits=3))
snaps={f:cpss_full(Z[val],(ysub[val]==ci).astype(int),seed=42+ci) for ci,f in enumerate(FAM_MC)}
sc=np.full((len(ev),6),-1e9)
for ci,f in enumerate(FAM_MC):
    pi,cf=snaps[f][15]['union']; s,_,_=cas_scores(Z[ev],pi,cf,0.8)
    if s is not None: sc[:,ci]=s
print('CAS (pi0.8,B15):'); print(classification_report(ysub[ev],np.argmax(sc,1),target_names=FAM_MC,zero_division=0,digits=3))
