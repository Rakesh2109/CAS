#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
for seed in 42 7 123; do
  nohup python3 scripts/full_train.py --prefix dedup --arm multiclass --seed $seed --epochs 15 > results/fulllogs/dd_mc_$seed.log 2>&1 &
  nohup python3 scripts/full_train.py --prefix dedup --arm binary --seed $seed --epochs 15 > results/fulllogs/dd_bin_$seed.log 2>&1 &
done
sleep 5
pgrep -af "prefix dedup" | grep -c full_train
