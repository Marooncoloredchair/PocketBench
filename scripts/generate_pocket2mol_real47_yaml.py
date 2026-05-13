#!/usr/bin/env python3
"""Download RCSB PDBs and infer ligand chain:resseq for the 47-pocket Colab run list; write YAML."""

from __future__ import annotations

import urllib.request
from collections import defaultdict
from pathlib import Path

from Bio.PDB import PDBParser, is_aa

from sbdd_robust.datasets.pocket_extraction import extract_pocket

POCKET_IDS = [
    "1AO7",
    "1B0R",
    "1DIZ",
    "1HXC",
    "1I4F",
    "1JF1",
    "1PVS",
    "1YFW",
    "1YFY",
    "2AV1",
    "2GJ6",
    "2P5E",
    "2P5W",
    "2PYE",
    "2X4N",
    "2X4O",
    "2X4Q",
    "2X4S",
    "2X4T",
    "3CWA",
    "3LZ9",
    "3M00",
    "3UPR",
    "3VRI",
    "3VRJ",
    "3WUW",
    "4HSJ",
    "4HSL",
    "4HVR",
    "4I3P",
    "4R52",
    "4RNQ",
    "4UQ3",
    "4WJ5",
    "4WZC",
    "5DHK",
    "5EAT",
    "5IK0",
    "5IK6",
    "5IKA",
    "5IL8",
    "5ILI",
    "5ILY",
    "5ILZ",
    "5JHD",
    "5T70",
    "5U98",
]

EXCLUDE_HET = {
    "HOH",
    "WAT",
    "DOD",
    "SO4",
    "PO4",
    "CL",
    "NA",
    "K",
    "MG",
    "CA",
    "BR",
    "IOD",
    "GOL",
    "EDO",
    "ACT",
    "DMS",
    "FMT",
    "PEG",
    "MPD",
    "TRS",
    "HEZ",
    "NO3",
    "NH2",
}

STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
    "ASX",
    "GLX",
    "UNK",
}


def fetch_pdb(pdb_id: str, dest: Path) -> None:
    pid = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    if not dest.is_file() or dest.stat().st_size < 1000:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def infer_from_hetatm_lines(pdb_path: Path):
    counts = defaultdict(int)
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if len(line) < 27:
                continue
            resname = line[17:20].strip().upper()
            if resname in EXCLUDE_HET or len(resname) < 2:
                continue
            chain = line[21:22].strip() or "A"
            try:
                seq = int(line[22:26])
            except ValueError:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem and len(line) >= 16:
                elem = line[12:14].strip().upper()[:1]
            if elem in ("H", "D", "", "Q"):
                continue
            counts[(chain, seq, resname)] += 1
    if not counts:
        return None
    rich = {k: v for k, v in counts.items() if v >= 3}
    pool = rich or counts
    (chain, seq, _resname), _n = max(pool.items(), key=lambda kv: kv[1])
    return chain, int(seq), f"{chain}:{int(seq)}"


def infer_from_atom_nonstandard(pdb_path: Path):
    counts = defaultdict(int)
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 27:
                continue
            resname = line[17:20].strip().upper()
            if resname in STANDARD_AA or resname in EXCLUDE_HET:
                continue
            chain = line[21:22].strip() or "A"
            try:
                seq = int(line[22:26])
            except ValueError:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem and len(line) >= 16:
                elem = line[12:14].strip().upper()[:1]
            if elem in ("H", "D", "", "Q"):
                continue
            counts[(chain, seq, resname)] += 1
    if not counts:
        return None
    rich = {k: v for k, v in counts.items() if v >= 5}
    pool = rich or counts
    (chain, seq, _rn), _n = max(pool.items(), key=lambda kv: kv[1])
    return chain, int(seq), f"{chain}:{int(seq)}"


def infer_ref_ligand(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    best = None
    best_n = -1
    for model in struct:
        for chain in model:
            cid = str(chain.id).strip() or "A"
            for res in chain:
                if is_aa(res, standard=True):
                    continue
                het = res.get_resname().strip().upper()
                if het in EXCLUDE_HET or len(het) <= 1:
                    continue
                n_heavy = 0
                for atom in res.get_atoms():
                    el = (atom.element or "").upper()
                    if el in ("H", "D", ""):
                        continue
                    n_heavy += 1
                if n_heavy > best_n:
                    best_n = n_heavy
                    rnum = int(res.id[1])
                    best = (cid, rnum, f"{cid}:{rnum}")
    if best is not None:
        return best
    hit = infer_from_hetatm_lines(pdb_path)
    if hit is not None:
        return hit
    hit = infer_from_atom_nonstandard(pdb_path)
    if hit is not None:
        return hit
    raise RuntimeError(f"No inferable ligand in {pdb_path.name}")


def pocket_entry(pid: str, ch, rs, ref, stem: str) -> dict:
    return {
        "id": pid,
        "pdb": f"data/raw/real50/{stem}.pdb",
        "ligand_chain": ch,
        "ligand_resseq": int(rs),
        "full_pdb": f"data/raw/real50/{stem}.pdb",
        "ref_ligand": ref,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "real50"
    raw.mkdir(parents=True, exist_ok=True)
    radius = 8.0
    pocket_rows: list[dict] = []
    for pid in POCKET_IDS:
        stem = pid.lower()
        dest = raw / f"{stem}.pdb"
        try:
            fetch_pdb(pid, dest)
            ch, rs, ref = infer_ref_ligand(dest)
            extract_pocket(
                dest,
                ligand_chain=str(ch),
                ligand_resseq=int(rs),
                radius=radius,
                pocket_id=pid,
            )
            pocket_rows.append(pocket_entry(pid, ch, rs, ref, stem))
            print("OK", pid, ref)
        except Exception as e:
            print("FAIL", pid, e)
            raise

    import yaml

    cfg = {
        "project_root": ".",
        "seed": 0,
        "paths": {"results": "data/results", "generations": "data/generations"},
        "extraction_radius": radius,
        "skip_failed_pockets": True,
        "brittleness_std_threshold": 0.05,
        "invariant_tags": [
            "atom_shuffle",
            "coordinate_jitter",
            "crop_radius_plus_1.5",
            "crop_radius_minus_1.5",
        ],
        "pockets": pocket_rows,
        "perturbations": [
            {"tag": "original", "type": "identity"},
            {"tag": "atom_shuffle", "type": "atom_shuffle", "seed": 0},
            {"tag": "coordinate_jitter", "type": "coordinate_jitter", "sigma": 0.1, "seed": 1},
            {"tag": "crop_radius_plus_1.5", "type": "crop_radius_plus", "delta_angstrom": 1.5},
            {"tag": "crop_radius_minus_1.5", "type": "crop_radius_minus", "delta_angstrom": 1.5},
        ],
        "model": {
            "type": "pocket2mol",
            "n_samples": 20,
            "repo_root": "D:/obsfu/Pocket2Mol",
            "checkpoint": "D:/obsfu/Pocket2Mol/ckpt/pretrained_Pocket2Mol.pt",
            "python_exe": "D:/Miniforge/envs/pocket2mol/python.exe",
            "script_path": "D:/obsfu/sbdd-robust/scripts/pocket2mol_sample_drug_bridge.py",
            "sanitize": True,
            "extra_args": [],
        },
    }
    out = root / "configs" / "pocket2mol_real47.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Wrote", out, "n_pockets=", len(pocket_rows))


if __name__ == "__main__":
    main()
