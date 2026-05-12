#!/usr/bin/env bash
# Placeholder: expand to full invariance suite (shuffle, jitter, crop, rename).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "01_invariance_benchmark: add configs/invariance.yaml and run sbdd-robust run --config ..."
