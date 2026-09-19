#!/usr/bin/env bash
# Submit (or dry-run) a PocketBench SLURM array job.
#
#   bash cluster/submit.sh --config configs/diffsbdd_real100.yaml --job-name pb-real100
#   bash cluster/submit.sh --config configs/diffsbdd_real100.yaml --job-name pb-real100 --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG=""
JOB_NAME="pocketbench"
DRY_RUN=0
EXTRA_SBATCH=()

usage() {
  cat <<'EOF'
Usage: cluster/submit.sh --config PATH --job-name NAME [--dry-run] [-- sbatch-args...]

Reads pocket count from YAML, writes a filled SLURM script under
$POCKETBENCH_WORK/slurm/, then sbatch (unless --dry-run).

Environment: source cluster/env.local.sh first (see env.example.sh).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_SBATCH+=("$@"); break ;;
    *) EXTRA_SBATCH+=("$1"); shift ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  usage >&2
  exit 2
fi

# Prefer env.local if present
if [[ -f "$SCRIPT_DIR/env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.local.sh"
fi

export POCKETBENCH_ROOT="${POCKETBENCH_ROOT:-$REPO_DIR}"
export POCKETBENCH_WORK="${POCKETBENCH_WORK:-$POCKETBENCH_ROOT/cluster_runs}"
export POCKETBENCH_RESULTS="${POCKETBENCH_RESULTS:-$POCKETBENCH_WORK/results}"
export POCKETBENCH_GENERATIONS="${POCKETBENCH_GENERATIONS:-$POCKETBENCH_WORK/generations}"

# Resolve config path (relative → repo)
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$POCKETBENCH_ROOT/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 1
fi
export POCKETBENCH_CONFIG="$CONFIG"

# Count pockets with a tiny Python helper (no site-specific deps beyond PyYAML/stdlib)
N_POCKETS="$(
  python3 - "$CONFIG" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
# Prefer PyYAML when available; fall back to a simple block count.
try:
    import yaml
    cfg = yaml.safe_load(text)
    pockets = cfg.get("pockets") or []
    print(len(pockets))
except Exception:
    # Count top-level "- id:" under pockets: (fragile but dry-run friendly)
    in_pockets = False
    n = 0
    for line in text.splitlines():
        if line.startswith("pockets:"):
            in_pockets = True
            continue
        if in_pockets:
            if line and not line.startswith(" ") and not line.startswith("-") and not line.startswith("#"):
                break
            if line.lstrip().startswith("- id:") or line.lstrip().startswith("-id:"):
                n += 1
    print(n)
PY
)"

if [[ -z "$N_POCKETS" || "$N_POCKETS" -lt 1 ]]; then
  echo "ERROR: could not count pockets in $CONFIG (got '$N_POCKETS')" >&2
  exit 1
fi

ARRAY_MAX=$((N_POCKETS - 1))
RUN_ID="$(
  python3 - "$CONFIG" <<'PY' || true
import sys
from pathlib import Path
try:
    import yaml
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
    print(cfg.get("run_id") or "")
except Exception:
    print("")
PY
)"
export POCKETBENCH_RUN_ID="${POCKETBENCH_RUN_ID:-${RUN_ID}}"

SLURM_DIR="$POCKETBENCH_WORK/slurm"
LOG_DIR="$POCKETBENCH_WORK/logs"
mkdir -p "$SLURM_DIR" "$LOG_DIR" "$POCKETBENCH_RESULTS" "$POCKETBENCH_GENERATIONS"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_OUT="$SLURM_DIR/${JOB_NAME}_${STAMP}.slurm"

ACCOUNT_LINE=""
if [[ -n "${POCKETBENCH_ACCOUNT:-}" ]]; then
  ACCOUNT_LINE="#SBATCH --account=${POCKETBENCH_ACCOUNT}"
fi

QOS_LINE=""
POCKETBENCH_QOS="${POCKETBENCH_QOS:-long}"
if [[ -n "$POCKETBENCH_QOS" ]]; then
  QOS_LINE="#SBATCH -q ${POCKETBENCH_QOS}"
fi

cat >"$SCRIPT_OUT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${POCKETBENCH_PARTITION:-uri-gpu}
${QOS_LINE}
${ACCOUNT_LINE}
#SBATCH --time=${POCKETBENCH_TIME:-7-00:00:00}
#SBATCH --gres=${POCKETBENCH_GRES:-gpu:${POCKETBENCH_GPUS:-1}}
#SBATCH --cpus-per-task=${POCKETBENCH_CPUS:-4}
#SBATCH --mem=${POCKETBENCH_MEM:-32G}
#SBATCH --array=0-${ARRAY_MAX}
#SBATCH --output=${LOG_DIR}/%x_%A_%a.out
#SBATCH --error=${LOG_DIR}/%x_%A_%a.err

set -euo pipefail
export POCKETBENCH_ROOT="${POCKETBENCH_ROOT}"
export POCKETBENCH_CONFIG="${POCKETBENCH_CONFIG}"
export POCKETBENCH_WORK="${POCKETBENCH_WORK}"
export POCKETBENCH_RESULTS="${POCKETBENCH_RESULTS}"
export POCKETBENCH_GENERATIONS="${POCKETBENCH_GENERATIONS}"
export POCKETBENCH_RUN_ID="${POCKETBENCH_RUN_ID}"
export POCKETBENCH_CONDA_ENV="${POCKETBENCH_CONDA_ENV:-diffsbdd}"
export POCKETBENCH_PYTHON="${POCKETBENCH_PYTHON:-}"
export POCKETBENCH_MODULES="${POCKETBENCH_MODULES:-}"
export POCKETBENCH_RESUME="${POCKETBENCH_RESUME:-1}"
export POCKETBENCH_NORMALIZED_ONLY="${POCKETBENCH_NORMALIZED_ONLY:-0}"
export DIFFSBDD_REPO="${DIFFSBDD_REPO:-}"
export DIFFSBDD_CHECKPOINT="${DIFFSBDD_CHECKPOINT:-}"
export DIFFSBDD_PYTHON="${DIFFSBDD_PYTHON:-}"
export POCKET2MOL_REPO="${POCKET2MOL_REPO:-}"
export POCKET2MOL_CHECKPOINT="${POCKET2MOL_CHECKPOINT:-}"
export POCKET2MOL_PYTHON="${POCKET2MOL_PYTHON:-}"
export VINA_EXE="${VINA_EXE:-}"

cd "\${POCKETBENCH_ROOT}"
bash cluster/run_array_task.sh
EOF

echo "=== PocketBench SLURM plan ==="
echo "site:        ${POCKETBENCH_SITE:-unset}"
echo "repo:        $POCKETBENCH_ROOT"
echo "config:      $POCKETBENCH_CONFIG"
echo "job-name:    $JOB_NAME"
echo "pockets:     $N_POCKETS  (array 0-${ARRAY_MAX})"
echo "run_id:      ${POCKETBENCH_RUN_ID:-<from CLI/timestamp>}"
echo "results:     $POCKETBENCH_RESULTS"
echo "generations: $POCKETBENCH_GENERATIONS"
echo "script:      $SCRIPT_OUT"
echo "partition:   ${POCKETBENCH_PARTITION:-uri-gpu}"
echo "qos:         ${POCKETBENCH_QOS:-}"
echo "time:        ${POCKETBENCH_TIME:-7-00:00:00}"
echo "gpus:        ${POCKETBENCH_GRES:-gpu:${POCKETBENCH_GPUS:-1}}"
echo "mem:         ${POCKETBENCH_MEM:-32G}"
if [[ ${#EXTRA_SBATCH[@]} -gt 0 ]]; then
  echo "extra:       ${EXTRA_SBATCH[*]}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "[dry-run] Not submitting. First 30 lines of script:"
  head -n 30 "$SCRIPT_OUT"
  exit 0
fi

sbatch "${EXTRA_SBATCH[@]}" "$SCRIPT_OUT"
