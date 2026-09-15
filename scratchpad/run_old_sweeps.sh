#!/bin/bash
cd /FPTM/CAS_IoMT_Empirical
pkill -f "full_cas.py --prefix old" 2>/dev/null || true
sleep 2
nohup python3 scripts/old_sweeps_final.py > results/old_sweeps_final.log 2>&1 &
sleep 5
pgrep -f old_sweeps_final.py | head -1
