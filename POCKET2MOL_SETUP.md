# Pocket2Mol integration (`Pocket2MolAdapter`)

## Repository and entrypoint

The historical **`luost26/Pocket2Mol`** GitHub org/repo is not available (404). Use the maintained code drop **`pengxingang/Pocket2Mol`**:

```bash
cd D:/obsfu
git clone https://github.com/pengxingang/Pocket2Mol.git Pocket2Mol
```

That tree does **not** include **`sample_drug.py`**. Inference for arbitrary PDB pockets is **`sample_for_pdb.py`**, driven by **`configs/sample_for_pdb.yml`** (checkpoint path, `num_samples`, beam search, etc.) plus **`--pdb_path`**, **`--center x,y,z`**, **`--bbox_size`**, **`--device`**, and **`--outdir`**.

`sbdd-robust` ships **`scripts/pocket2mol_sample_drug_bridge.py`**, which exposes the adapter’s expected flags (`--pdb_path`, `--num_samples`, `--result_path`, `--checkpoint`), infers **center** and **bbox** from the pocket PDB, runs **`sample_for_pdb.py`**, and copies **`samples_all.pt`** to **`--result_path`**.

In YAML, set:

```yaml
model:
  type: pocket2mol
  repo_root: D:/obsfu/Pocket2Mol
  script_path: D:/obsfu/sbdd-robust/scripts/pocket2mol_sample_drug_bridge.py
  checkpoint: D:/obsfu/Pocket2Mol/ckpt/pretrained_Pocket2Mol.pt
  python_exe: D:/Miniforge/envs/pocket2mol/python.exe
```

Optional **`model.extra_args`** are forwarded to **`sample_for_pdb.py`** (e.g. `["--device", "cpu"]`). The bridge also accepts **`--device`** as its own first-class flag when invoked manually.

## Checkpoint weights

See **`Pocket2Mol/ckpt/README.md`**: the pretrained file is distributed via **Google Drive** (filename along the lines of **`pretrained_Pocket2Mol.pt`**). Download it into **`D:/obsfu/Pocket2Mol/ckpt/`** so it matches **`configs/sample_for_pdb.yml`** / the **`model.checkpoint`** path in your **`sbdd-robust`** YAML.

## Conda environment (example)

Pocket2Mol depends on **PyTorch Geometric** and versions that differ from DiffSBDD. Use a dedicated env, e.g.:

```bash
conda create -n pocket2mol python=3.8
conda activate pocket2mol
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric torch-scatter torch-sparse
pip install rdkit biopython tqdm pyyaml easydict
```

Or follow **`env_cuda113.yml`** in the Pocket2Mol repo. Point **`model.python_exe`** at that interpreter.

## Standalone smoke test

```bash
conda activate pocket2mol
cd D:/obsfu/Pocket2Mol
# after placing the checkpoint under ckpt/
python sample_for_pdb.py --pdb_path example/4yhj.pdb --config configs/sample_for_pdb.yml --device cuda --outdir ./outputs
```

Bridge smoke test (writes a single **`test_out.pt`**):

```bash
python D:/obsfu/sbdd-robust/scripts/pocket2mol_sample_drug_bridge.py ^
  --pdb_path D:/obsfu/sbdd-robust/data/raw/real50/1ao7.pdb ^
  --num_samples 5 ^
  --result_path D:/obsfu/tmp/test_out.pt ^
  --checkpoint D:/obsfu/Pocket2Mol/ckpt/pretrained_Pocket2Mol.pt ^
  --device cuda
```

## 47-pocket benchmark config

**`configs/experiments/pocket2mol_real47.yaml`** mirrors the Colab DiffSBDD **real50** run (same **47** PDB IDs as export **run1778615375**, **8 Å** extraction, **0.05** brittleness threshold, **σ = 0.1** coordinate jitter, same invariant **tags**). PDBs live under **`data/raw/real50/`** (downloaded when you run **`scripts/generate_pocket2mol_real47_yaml.py`**).

```bash
cd D:/obsfu/sbdd-robust
conda activate pocket2mol
python -m sbdd_robust run --config configs/experiments/pocket2mol_real47.yaml
```

Use the **base** `python` for **`python -m sbdd_robust`** if you like; **`model.python_exe`** is only used for the Pocket2Mol subprocess chain.

## What the adapter consumes

The bridge leaves **`samples_all.pt`** (EasyDict **`pool`** with **`finished`** molecules). **`Pocket2MolAdapter`** loads that payload and converts entries to RDKit **`Mol`** objects (SMILES / **`rdmol`** / coordinate reconstruction as implemented in **`pocket2mol_adapter.py`**).
