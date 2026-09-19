#!/bin/bash
set +e
WORK=/work/pi_nzawia_uri_edu/pocketbench
cd "$WORK/repo"
echo "=== smoke metrics ==="
cat data/results/metrics_per_condition__runp2m_env_smoke.csv
echo "=== smoke gens under repo ==="
find data/generations/run_p2m_env_smoke -type f 2>/dev/null | head
echo "=== smoke gens under runs ==="
find "$WORK/runs/generations" -iname '*p2m_env_smoke*' -type f 2>/dev/null | head
find "$WORK/runs" -iname '*p2m_env_smoke*' 2>/dev/null | head
echo "=== last P2M out dir leftovers? ==="
ls -la data/generations/run_p2m_env_smoke 2>/dev/null
echo INSPECT_DONE
