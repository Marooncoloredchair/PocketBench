#!/usr/bin/env bash
# Environment template for PocketBench on SLURM.
# Copy to env.local.sh (gitignored) and edit. Source before submit.sh.
#
#   cp cluster/env.example.sh cluster/env.local.sh
#   source cluster/env.local.sh

# --- site ---
# unity | epfl
export POCKETBENCH_SITE="${POCKETBENCH_SITE:-unity}"

# --- repo & scratch (POSIX only; no Windows paths) ---
export POCKETBENCH_ROOT="${POCKETBENCH_ROOT:-/work/pi_nzawia_uri_edu/pocketbench/repo}"
export POCKETBENCH_WORK="${POCKETBENCH_WORK:-/work/pi_nzawia_uri_edu/pocketbench/runs}"
export POCKETBENCH_RESULTS="${POCKETBENCH_RESULTS:-${POCKETBENCH_WORK}/results}"
export POCKETBENCH_GENERATIONS="${POCKETBENCH_GENERATIONS:-${POCKETBENCH_WORK}/generations}"

# --- conda / python ---
export POCKETBENCH_CONDA_ENV="${POCKETBENCH_CONDA_ENV:-diffsbdd}"
export POCKETBENCH_PYTHON="${POCKETBENCH_PYTHON:-}"

# --- CUDA / modules (Unity example; override on EPFL) ---
export POCKETBENCH_MODULES="${POCKETBENCH_MODULES:-cuda/12.6}"

# --- model checkpoints (absolute POSIX paths) ---
export DIFFSBDD_REPO="${DIFFSBDD_REPO:-/work/pi_nzawia_uri_edu/pocketbench/DiffSBDD}"
export DIFFSBDD_CHECKPOINT="${DIFFSBDD_CHECKPOINT:-/work/pi_nzawia_uri_edu/pocketbench/ckpts/crossdocked_fullatom_cond.ckpt}"
# Optional; leave empty to use the conda env's python
export DIFFSBDD_PYTHON="${DIFFSBDD_PYTHON:-}"
export POCKET2MOL_REPO="${POCKET2MOL_REPO:-/work/pi_nzawia_uri_edu/pocketbench/Pocket2Mol}"
export POCKET2MOL_CHECKPOINT="${POCKET2MOL_CHECKPOINT:-/work/pi_nzawia_uri_edu/pocketbench/ckpts/pretrained_Pocket2Mol.pt}"
# Dedicated Pocket2Mol env — do NOT point this at the pocketbench interpreter.
# See cluster/README.md "Conda environments (two, not one)" and cluster/build_p2m_env.sh.
export POCKET2MOL_PYTHON="${POCKET2MOL_PYTHON:-/work/pi_nzawia_uri_edu/pocketbench/envs/p2m/bin/python}"

# --- SLURM defaults (Unity) ---
export POCKETBENCH_PARTITION="${POCKETBENCH_PARTITION:-gpu}"
export POCKETBENCH_QOS="${POCKETBENCH_QOS:-long}"
export POCKETBENCH_TIME="${POCKETBENCH_TIME:-7-00:00:00}"
export POCKETBENCH_GPUS="${POCKETBENCH_GPUS:-1}"
# Default: any GPU. Pocket2Mol MUST override this — Unity's gpu partition
# includes Tesla M40 (sm_52 / Maxwell) and CUDA 12 cannot target Maxwell
# (cudaErrorNoKernelImageForDevice from torch-cluster). Job B sets
# POCKETBENCH_GRES=gpu:2080_ti:1 (Turing sm_75).
export POCKETBENCH_GRES="${POCKETBENCH_GRES:-gpu:${POCKETBENCH_GPUS:-1}}"
export POCKETBENCH_CPUS="${POCKETBENCH_CPUS:-4}"
export POCKETBENCH_MEM="${POCKETBENCH_MEM:-32G}"
# EPFL: often also export POCKETBENCH_ACCOUNT=…

# --- optional docking (Job A) ---
export VINA_EXE="${VINA_EXE:-}"
