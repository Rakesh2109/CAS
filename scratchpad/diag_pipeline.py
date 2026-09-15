import os, time, sys, json
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0,"/tmp/claude-0/-FPTM/4822c625-085b-4d7c-bfe0-51629b5ba47d/scratchpad/tmu_repo")
from tmu.models.classification.vanilla_classifier import TMClassifier

# 1) paper's own 22-feat/10-bin booleanized data, my training loop, paper config
d=np.load("results/booleanized_data.npz")
Xtr,ytr,Xte,yte=d["Xtr"],d["ytr"],d["Xte"],d["yte"]
print("paper 22-feat data:",Xtr.shape, "-> C=400 T=320 s=8 x25ep")
tm=TMClassifier(number_of_clauses=400,T=320,s=8.0,weighted_clauses=True,platform="CPU",seed=42)
for ep in range(1,26):
    tm.fit(Xtr,ytr)
    if ep%5==0:
        f1=f1_score(yte,tm.predict(Xte),average="macro")
        print(f"  ep{ep}: macro-F1={f1:.4f}",flush=True)
