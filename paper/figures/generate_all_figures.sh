#!/usr/bin/env bash
# Regenerate paper figures from archived CSVs (run from repo root: bash paper/figures/generate_all_figures.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
METRICS="${SBDD_ROBUST_METRICS_CSV:-$ROOT/data/results/metrics_per_condition__run1778615375.csv}"
P2="${SBDD_ROBUST_P2_METRICS:-$ROOT/data/results/metrics_per_condition__runpocket2mol_real47_full.csv}"

python analysis/compare_models.py --diffsbdd "$METRICS" --pocket2mol "$P2"
python analysis/threshold_sensitivity.py --metrics "$METRICS"
python analysis/dominant_brittleness_metric.py --metrics "$METRICS" || true
python analysis/nsamples_sensitivity.py || true
python paper/figures/fig3_sa_drift.py --metrics "$METRICS"
python paper/figures/fig2_3rfm_validity.py || true
python scripts/run_consistency_filter_real47.py --metrics "$METRICS"
python paper/figures/fig_consistency_filter.py
python analysis/meaningful_perturbation_analysis.py || true
echo "Done. PDF+PNG emitted where scripts use nmi_style.save_figure."
