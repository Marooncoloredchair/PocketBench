#!/usr/bin/env bash
# One SLURM array task: run a single pocket from the YAML panel.
set -euo pipefail

ROOT="${POCKETBENCH_ROOT:?POCKETBENCH_ROOT unset}"
CFG="${POCKETBENCH_CONFIG:?POCKETBENCH_CONFIG unset}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset (not an array task?)}"

cd "$ROOT"

# Lmod (Unity batch jobs are not login shells)
if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  set +u
  # shellcheck source=/dev/null
  source /usr/share/lmod/lmod/init/bash
  set -u
fi

if [[ -n "${POCKETBENCH_MODULES:-}" ]]; then
  module purge 2>/dev/null || true
  # shellcheck disable=SC2086
  module load ${POCKETBENCH_MODULES}
fi

if [[ -n "${POCKETBENCH_PYTHON:-}" ]]; then
  PY="$POCKETBENCH_PYTHON"
elif command -v conda >/dev/null 2>&1 && [[ -n "${POCKETBENCH_CONDA_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$POCKETBENCH_CONDA_ENV"
  PY=python
else
  PY=python
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSBDD_REPO="${DIFFSBDD_REPO:-}"
export DIFFSBDD_CHECKPOINT="${DIFFSBDD_CHECKPOINT:-}"
export VINA_EXE="${VINA_EXE:-}"

EXTRA=()
[[ -n "${POCKETBENCH_RUN_ID:-}" ]] && EXTRA+=(--run-id "$POCKETBENCH_RUN_ID")
[[ "${POCKETBENCH_RESUME:-1}" == "1" ]] && EXTRA+=(--resume)
[[ "${POCKETBENCH_NORMALIZED_ONLY:-0}" == "1" ]] && EXTRA+=(--normalized-only)

echo "[pocketbench] host=$(hostname) task=$TASK_ID cfg=$CFG py=$PY"
exec "$PY" -m sbdd_robust.cli run \
  --config "$CFG" \
  --pocket-index "$TASK_ID" \
  "${EXTRA[@]}"
