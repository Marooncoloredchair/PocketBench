#!/usr/bin/env bash
# Run paper reproduction helpers in order (table/fig from metrics; optional smoke test).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  SBDD-Robust — reproduce_all (paper artifacts + smoke)"
echo "============================================================"

echo ""
echo ">>> [1/3] reproduce_table1.sh (Table 1 + threshold curve from metrics CSV)"
bash "$ROOT/experiments/reproduce_table1.sh"

echo ""
echo ">>> [2/3] reproduce_fig1.sh (Fig. 1 PDF; same data as table step)"
bash "$ROOT/experiments/reproduce_fig1.sh"

echo ""
echo ">>> [3/3] 00_smoke_test.sh (mock model CI-style check)"
bash "$ROOT/experiments/00_smoke_test.sh"

echo ""
echo "============================================================"
echo "  reproduce_all finished."
echo "============================================================"
