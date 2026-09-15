#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
while [ "$(grep -laE 'FINAL' results/fulllogs/*.log 2>/dev/null | wc -l)" -lt 6 ]; do
  n=$(pgrep -cf "scripts/full_train.py")
  [ "$n" -eq 0 ] && break
  sleep 60
done
echo "TRAINING DONE $(date)"
grep -aE "FINAL" results/fulllogs/*.log | grep -av pycuda
nohup python3 scripts/full_cas.py --arm multiclass --seeds 42,7,123 > results/full_cas_mc.log 2>&1 &
nohup python3 scripts/full_cas.py --arm binary --seeds 42,7,123 > results/full_cas_bin.log 2>&1 &
echo "full_cas launched (mc + bin)"
