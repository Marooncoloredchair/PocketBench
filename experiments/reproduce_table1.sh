#!/usr/bin/env bash
# Reproduce paper Table 1 (DiffSBDD brittleness vs τ on 47 covered pockets) and Fig. 1 curve.
#
# Default: recompute from frozen metrics (exact paper numbers, seconds on CPU).
# Optional full regeneration: SBDD_ROBUST_FULL_RERUN=1 and DiffSBDD env vars (days of GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METRICS_CSV="${SBDD_ROBUST_METRICS_CSV:-$ROOT/data/results/metrics_per_condition__run1778615375.csv}"
OUT_CSV="${SBDD_ROBUST_TABLE1_CSV:-$ROOT/paper/threshold_sensitivity.csv}"
OUT_FIG="${SBDD_ROBUST_FIG1_PDF:-$ROOT/paper/figures/threshold_sensitivity.pdf}"

if [[ "${SBDD_ROBUST_FULL_RERUN:-0}" == "1" ]]; then
  echo "==> [reproduce_table1] Full DiffSBDD 47-pocket run (long). Checking env..."
  : "${DIFFSBDD_REPO:?Export DIFFSBDD_REPO (DiffSBDD clone root)}"
  : "${DIFFSBDD_CHECKPOINT:?Export DIFFSBDD_CHECKPOINT (path to .ckpt)}"
  : "${DIFFSBDD_PYTHON:?Export DIFFSBDD_PYTHON (python in DiffSBDD env)}"
  echo "==> [reproduce_table1] Running: python -m sbdd_robust run --config configs/experiments/diffsbdd_real47.yaml"
  python -m sbdd_robust run --config "$ROOT/configs/experiments/diffsbdd_real47.yaml"
  METRICS_CSV="$ROOT/data/results/metrics_per_condition__rundiffsbdd_real47.csv"
fi

if [[ ! -f "$METRICS_CSV" ]]; then
  echo "ERROR: metrics CSV not found: $METRICS_CSV"
  exit 1
fi

echo "==> [reproduce_table1] Threshold sensitivity → $OUT_CSV , $OUT_FIG"
python analysis/threshold_sensitivity.py \
  --metrics "$METRICS_CSV" \
  --out-csv "$OUT_CSV" \
  --out-figure "$OUT_FIG"

echo "==> [reproduce_table1] Done."
