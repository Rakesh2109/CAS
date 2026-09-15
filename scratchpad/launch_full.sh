#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
pkill -9 -f "scripts/full_train.py" 2>/dev/null || true
sleep 3
for seed in 42 7 123; do
  nohup python3 scripts/full_train.py --arm multiclass --seed $seed --epochs 15 > results/fulllogs/mc_$seed.log 2>&1 &
  nohup python3 scripts/full_train.py --arm binary --seed $seed --epochs 15 > results/fulllogs/bin_$seed.log 2>&1 &
done
sleep 5
pgrep -af "scripts/full_train.py" | grep -c full_train
