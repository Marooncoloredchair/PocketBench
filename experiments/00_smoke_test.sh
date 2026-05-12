#!/usr/bin/env bash
# Smoke: 3 pockets, 2 invariant perturbations + original, mock model, full CSV + figures.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pip install -q -e ".[dev]"
python -m pytest tests/ -q
python -m sbdd_robust run --config configs/smoke.yaml
echo "Smoke OK. Inspect data/results/ for CSVs and figures."
