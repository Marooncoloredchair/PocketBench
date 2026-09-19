#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
P2M_ENV=$WORK/envs/p2m
REPO=$WORK/repo
cd "$REPO"

# One-pocket smoke config: separate run_id so it never touches the real panel.
cat > configs/cluster/_p2m_env_smoke.yaml <<'EOS'
project_root: .
seed: 0
run_id: p2m_env_smoke
paths:
  results: data/results
  generations: data/generations
extraction_radius: 8.0
skip_failed_pockets: false
brittleness_std_threshold: 0.05
invariant_tags:
- atom_shuffle
pockets:
- id: 1AO7
  pdb: data/raw/real50/1ao7.pdb
  ligand_chain: B
  ligand_resseq: 100
  full_pdb: data/raw/real50/1ao7.pdb
  ref_ligand: B:100
perturbations:
- tag: original
  type: identity
model:
  type: pocket2mol
  n_samples: 4
  repo_root: ${env:POCKET2MOL_REPO}
  checkpoint: ${env:POCKET2MOL_CHECKPOINT}
  python_exe: ${env:POCKET2MOL_PYTHON}
  script_path: scripts/pocket2mol_sample_drug_bridge.py
  sanitize: true
  extra_args:
  - --base_config
  - configs/sample_for_pdb_4gb.yml
resume: false
normalized_only: true
EOS

INNER=$WORK/runs/_p2m_smoke_inner.sh
mkdir -p "$WORK/runs"
cat > "$INNER" <<'EOS'
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
module load cuda/12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/pocketbench"
cd "$WORK/repo"
source cluster/env.local.sh
# Dedicated Pocket2Mol interpreter (NOT the pocketbench env)
export POCKET2MOL_PYTHON="$WORK/envs/p2m/bin/python"
export POCKET2MOL_REPO="$WORK/Pocket2Mol"
export POCKET2MOL_CHECKPOINT="$WORK/ckpts/pretrained_Pocket2Mol.pt"
echo "host=$(hostname)"
nvidia-smi -L || true
echo "P2M python: $POCKET2MOL_PYTHON"
"$POCKET2MOL_PYTHON" - <<'PY'
import torch
from torch_cluster import knn
print("p2m torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    x = torch.randn(50, 3, device="cuda")
    y = torch.randn(10, 3, device="cuda")
    print("knn", tuple(knn(x, y, k=6).shape))
PY
rm -f data/results/metrics_per_condition__runp2m_env_smoke*.csv
python -m sbdd_robust run --config configs/cluster/_p2m_env_smoke.yaml
echo "--- resulting molecules ---"
"$WORK/envs/p2m/bin/python" - <<'PY'
from pathlib import Path
import pandas as pd
csv = Path("data/results/metrics_per_condition__runp2m_env_smoke.csv")
df = pd.read_csv(csv)
print(df.to_string(index=False))
n = int(df["n_valid"].sum()) if not df.empty and "n_valid" in df.columns else 0
smiles = list(Path("data/generations/run_p2m_env_smoke").rglob("*_smiles.txt"))
print("smiles_files", [str(p) for p in smiles])
for p in smiles:
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(p.name, "n_smiles", len(lines))
print("TOTAL_VALID", n)
print("SMOKE_RESULT", "PASS" if n > 0 else "FAIL")
raise SystemExit(0 if n > 0 else 1)
PY
EOS

echo "=== running one-pocket P2M smoke on a GPU node ==="
srun --partition=gpu -q short --account=pi_nzawia_uri_edu --time=00:40:00 \
  --gres=gpu:2080_ti:1 --cpus-per-task=4 --mem=24G \
  bash "$INNER"
echo SMOKE_DONE
