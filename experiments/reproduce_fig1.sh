#!/usr/bin/env bash
# Regenerate Fig. 1 (threshold sensitivity PDF) from frozen or custom metrics CSV.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METRICS_CSV="${SBDD_ROBUST_METRICS_CSV:-$ROOT/data/results/metrics_per_condition__run1778615375.csv}"
OUT_CSV="$ROOT/paper/threshold_sensitivity.csv"
OUT_FIG="$ROOT/paper/figures/threshold_sensitivity.pdf"

echo "==> [reproduce_fig1] $METRICS_CSV → $OUT_FIG"
python analysis/threshold_sensitivity.py \
  --metrics "$METRICS_CSV" \
  --out-csv "$OUT_CSV" \
  --out-figure "$OUT_FIG"
echo "==> [reproduce_fig1] Done."
