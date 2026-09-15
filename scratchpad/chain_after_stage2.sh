#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
while ps -p 2918189 >/dev/null 2>&1; do sleep 60; done
echo "STAGE2 DONE $(date)"
nohup python3 scripts/gridsearch_tm22.py > results/gridsearch_tm22_log.txt 2>&1 &
echo "22feat grid started pid $!"
