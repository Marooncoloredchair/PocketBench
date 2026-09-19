#!/bin/bash
set -euo pipefail
echo "=== squeue ==="
squeue -u "$USER" | head -30
echo "=== Job A completed shards ==="
ls /work/pi_nzawia_uri_edu/pocketbench/runs/results/metrics_per_condition__rundiffsbdd_real100_vina__p*.csv 2>/dev/null | wc -l
echo "=== SDF count ==="
find /work/pi_nzawia_uri_edu/pocketbench/repo/data/generations/run_diffsbdd_real100_vina -name '*_gen.sdf' 2>/dev/null | wc -l
echo "=== recent A errs (fail signals) ==="
grep -lE 'ModuleNotFoundError|unrecognized arguments|RuntimeError: No benchmark' /work/pi_nzawia_uri_edu/pocketbench/runs/logs/pb-real100-vina_62848168_*.err 2>/dev/null | wc -l || echo 0
echo "=== recent A running/done sample ==="
tail -n 15 /work/pi_nzawia_uri_edu/pocketbench/runs/logs/pb-real100-vina_62848168_0.err 2>/dev/null || true
echo CHECK_DONE
