#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
nohup python3 scripts/pruned_cas.py > results/pruned_cas.log 2>&1 &
sleep 5
pgrep -f pruned_cas.py | head -1
