#!/usr/bin/env bash
#SBATCH --job-name=pocketbench
#SBATCH --partition=uri-gpu
#SBATCH -q long
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Reference template. Prefer: bash cluster/submit.sh --config … --job-name …
# which fills array bounds, paths, and env from cluster/env.local.sh.

set -euo pipefail

: "${POCKETBENCH_ROOT:?Set POCKETBENCH_ROOT (repo clone on the cluster)}"
: "${POCKETBENCH_CONFIG:?Set POCKETBENCH_CONFIG (YAML path)}"

cd "${POCKETBENCH_ROOT}"
# shellcheck source=/dev/null
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh

bash cluster/run_array_task.sh
