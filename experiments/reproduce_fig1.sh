#!/usr/bin/env bash
# Regenerate Fig. 1 (threshold sensitivity PDF) from frozen or custom metrics CSV.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METRICS_CSV="${SBDD_ROBUST_METRICS_CSV:-$ROOT/data/results/metrics_per_condition__run1778615375.csv}"
OUT_CSV="$ROOT/paper/threshold_sensitivity.csv"
OUT_FIG="$ROOT/paper/figures/threshold_sensitivity.pdf"

POCKET2MOL_CSV="${SBDD_ROBUST_POCKET2MOL_METRICS_CSV:-$ROOT/data/results/metrics_per_condition__runpocket2mol_real47_full.csv}"
echo "==> [reproduce_fig1] $METRICS_CSV → $OUT_FIG"
python analysis/threshold_sensitivity.py \
  --metrics "$METRICS_CSV" \
  --pocket2mol-metrics "$POCKET2MOL_CSV" \
  --out-csv "$OUT_CSV" \
  --out-figure "$OUT_FIG"
echo "==> [reproduce_fig1] Done."
