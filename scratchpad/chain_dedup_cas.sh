#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
while [ "$(grep -laE 'FINAL' results/fulllogs/dd_*.log 2>/dev/null | wc -l)" -lt 6 ]; do
  pgrep -f "prefix dedup" >/dev/null 2>&1 || break
  sleep 45
done
echo "DEDUP TRAINING DONE $(date)"
grep -aE "FINAL" results/fulllogs/dd_*.log | grep -av pycuda
nohup python3 scripts/full_cas.py --prefix dedup --arm multiclass --seeds 42,7,123 > results/dedup_cas_mc.log 2>&1 &
nohup python3 scripts/full_cas.py --prefix dedup --arm binary --seeds 42,7,123 > results/dedup_cas_bin.log 2>&1 &
echo "dedup full_cas launched"
