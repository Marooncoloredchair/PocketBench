#!/usr/bin/env python3
"""
Build configs/diffsbdd_real100.yaml and configs/pocket2mol_real100.yaml with 100 validated pockets.

Selection pipeline (network required):
1. Paginate RCSB Search API v2 for X-ray structures with reported resolution <= 2.5 Å.
2. Download mmCIF or PDB, reject if ligand residue MW (heavy atoms, periodic table) not in [150, 600] Da.
3. Deduplicate by primary UniProt accession per structure where available (rcsb.org data API).
4. Infer reference ligand (same heuristics as generate_pocket2mol_real47_yaml).
5. Call extract_pocket() to ensure the pocket tensor is non-degenerate.

Output paths use data/raw/real100/{pdb}.pdb and match diffsbdd_real47 perturbation / model blocks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from Bio.PDB import PDBParser, is_aa
from rdkit import Chem

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.datasets.pocket_extraction import extract_pocket

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/"
PDB_DOWNLOAD = "https://files.rcsb.org/download/{}.pdb"

EXCLUDE_HET = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "BR", "IOD",
    "GOL", "EDO", "ACT", "DMS", "FMT", "PEG", "MPD", "TRS", "HEZ", "NO3", "NH2", "UNL",
}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "SEC", "PYL", "ASX", "GLX", "UNK",
}


def _fetch_json(url: str, data: bytes | None = None, headers: dict | None = None) -> dict:
    hdr = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdr.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def uniprot_accessions(pdb_id: str) -> set[str]:
    """Return UniProt accessions declared for polymer entities in the entry."""
    pid = pdb_id.strip().lower()
    acc: set[str] = set()
    try:
        url = f"{RCSB_DATA_ENTRY}{pid}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return acc
    for ent in blob.get("polymer_entities", []) or []:
        for ref in ent.get("entity_poly", {}).get("pdbx_strand_id", []):
            pass
        xids = ent.get("rcsb_polymer_entity_container_identifiers", {})
        for it in xids.get("reference_sequence_identifiers", []) or []:
            if str(it.get("database_name", "")).upper() == "UNIPROT":
                da = it.get("database_accession")
                if da:
                    acc.add(str(da).strip())
    return acc


def resolution_from_api(pdb_id: str) -> float | None:
    pid = pdb_id.strip().lower()
    try:
        req = urllib.request.Request(
            RCSB_DATA_ENTRY + pid, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            blob = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None
    ri = blob.get("rcsb_entry_info", {})
    r = ri.get("resolution_combined")
    if r is None:
        return None
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def rcsb_search_ids(*, rows: int = 500, start: int = 0) -> tuple[list[str], int | None]:
    """Return PDB IDs from RCSB Search API v2 (X-ray, resolution <= 2.5 Å)."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [
                {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}
            ],
        },
    }
    body = json.dumps(query).encode()
    try:
        out = _fetch_json(RCSB_SEARCH, data=body)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        print("WARNING: RCSB search failed:", e, file=sys.stderr)
        return [], None
    total_count = out.get("total_count")
    ids: list[str] = []
    for g in out.get("result_set", []):
        ident = g.get("identifier")
        if ident:
            ids.append(str(ident).upper())
    return ids, total_count


def fetch_pdb(pdb_id: str, dest: Path) -> None:
    pid = pdb_id.lower()
    url = PDB_DOWNLOAD.format(pid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _periodic_weights() -> dict[str, float]:
    pt = Chem.GetPeriodicTable()
    sym = (
        "H",
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "Fe",
        "Zn",
        "Cu",
        "Mn",
        "Mg",
        "Na",
        "K",
        "Ca",
        "Se",
    )
    return {el: pt.GetAtomicWeight(el) for el in sym}


_WTS = _periodic_weights()
_WTS_UPPER = {k.upper(): v for k, v in _WTS.items()}


def hetatm_mw_da(pdb_path: Path, chain: str, resseq: int, resname: str) -> float:
    mw = 0.0
    chain = chain.strip() or "A"
    resname = resname.strip().upper()
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("HETATM", "ATOM  ")):
                continue
            if len(line) < 27:
                continue
            rn = line[17:20].strip().upper()
            if rn != resname:
                continue
            ch = line[21:22].strip() or "A"
            if ch != chain:
                continue
            try:
                seq = int(line[22:26])
            except ValueError:
                continue
            if seq != resseq:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem:
                elem = line[12:14].strip().upper()
            elem = elem[:2] if len(elem) > 1 and elem[:2] in _WTS_UPPER else elem[:1]
            if elem in ("H", "D", ""):
                continue
            mw += float(_WTS_UPPER.get(elem.upper(), _WTS["C"]))
    return mw


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
                    best = (cid, rnum, het, f"{cid}:{rnum}")
    if best is not None:
        return best
    raise RuntimeError(f"No inferable ligand in {pdb_path.name}")


def pocket_entry(pid: str, ch, rs, ref: str, stem: str) -> dict:
    return {
        "id": pid,
        "pdb": f"data/raw/real100/{stem}.pdb",
        "ligand_chain": ch,
        "ligand_resseq": int(rs),
        "full_pdb": f"data/raw/real100/{stem}.pdb",
        "ref_ligand": ref,
    }


def write_yaml(path: Path, cfg: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=100, help="Number of pockets to collect")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mw-min", type=float, default=150.0)
    ap.add_argument("--mw-max", type=float, default=600.0)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--skip-rcsb-search", action="store_true", help="Use --id-file only")
    ap.add_argument(
        "--id-file",
        type=Path,
        default=None,
        help="Optional newline-separated PDB IDs to try (supplements or replaces search)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    raw = _ROOT / "data" / "raw" / "real100"
    raw.mkdir(parents=True, exist_ok=True)

    candidate_ids: list[str] = []
    if args.id_file and args.id_file.is_file():
        for line in args.id_file.read_text(encoding="utf-8").splitlines():
            s = line.strip().upper()
            if len(s) >= 4:
                candidate_ids.append(s[:4])

    if not args.skip_rcsb_search:
        start = 0
        total = None
        while True:
            batch, total = rcsb_search_ids(rows=500, start=start)
            candidate_ids.extend(batch)
            if total is not None and start + len(batch) >= total:
                break
            if not batch:
                break
            start += len(batch)
            if start > 15000:
                break
            time.sleep(0.3)

    candidate_ids = list(dict.fromkeys(candidate_ids))
    rng.shuffle(candidate_ids)

    seen_uniprot: set[str] = set()
    pocket_rows: list[dict] = []
    mw_lo, mw_hi = args.mw_min, args.mw_max

    for pdb_id in candidate_ids:
        if len(pocket_rows) >= args.target:
            break
        if resolution_from_api(pdb_id) is not None:
            r = resolution_from_api(pdb_id)
            if r is not None and r > 2.5:
                continue
        ups = uniprot_accessions(pdb_id)
        if ups:
            if any(u in seen_uniprot for u in ups):
                continue

        stem = pdb_id.lower()
        dest = raw / f"{stem}.pdb"
        try:
            fetch_pdb(pdb_id, dest)
            ch, rs, resname, ref = infer_ref_ligand(dest)
            mw = hetatm_mw_da(dest, ch, rs, resname)
            if not (mw_lo <= mw <= mw_hi):
                continue
            extract_pocket(
                dest,
                ligand_chain=str(ch),
                ligand_resseq=int(rs),
                radius=args.radius,
                pocket_id=pdb_id,
            )
            if ups:
                seen_uniprot.update(ups)
            pocket_rows.append(pocket_entry(pdb_id, ch, rs, ref, stem))
            print("OK", len(pocket_rows), pdb_id, ref, f"MW~{mw:.1f}")
        except Exception as e:
            print("SKIP", pdb_id, e, file=sys.stderr)
            continue

    print(f"Collected {len(pocket_rows)} / {args.target} pockets.")

    perturbations = [
        {"tag": "original", "type": "identity"},
        {"tag": "atom_shuffle", "type": "atom_shuffle", "seed": 0},
        {"tag": "coordinate_jitter", "type": "coordinate_jitter", "sigma": 0.1, "seed": 1},
        {"tag": "crop_radius_plus_1.5", "type": "crop_radius_plus", "delta_angstrom": 1.5},
        {"tag": "crop_radius_minus_1.5", "type": "crop_radius_minus", "delta_angstrom": 1.5},
    ]
    inv_tags = [
        "atom_shuffle",
        "coordinate_jitter",
        "crop_radius_plus_1.5",
        "crop_radius_minus_1.5",
    ]

    base = {
        "project_root": ".",
        "seed": 0,
        "run_id": "diffsbdd_real100",
        "paths": {"results": "data/results", "generations": "data/generations"},
        "extraction_radius": args.radius,
        "skip_failed_pockets": True,
        "brittleness_std_threshold": 0.05,
        "invariant_tags": inv_tags,
        "pockets": pocket_rows,
        "perturbations": perturbations,
    }
    diff_cfg = {
        **base,
        "model": {
            "type": "diffsbdd",
            "n_samples": 20,
            "repo_root": "${env:DIFFSBDD_REPO}",
            "checkpoint": "${env:DIFFSBDD_CHECKPOINT}",
            "python_exe": "${env:DIFFSBDD_PYTHON}",
            "resamplings": 10,
            "jump_length": 1,
            "sanitize": False,
        },
    }
    p2_cfg = {
        **base,
        "run_id": "pocket2mol_real100",
        "model": {
            "type": "pocket2mol",
            "n_samples": 20,
            "repo_root": "${env:POCKET2MOL_REPO}",
            "checkpoint": "${env:POCKET2MOL_CHECKPOINT}",
            "python_exe": "${env:POCKET2MOL_PYTHON}",
            "script_path": "${env:POCKET2MOL_SCRIPT}",
            "sanitize": True,
            "extra_args": [],
        },
    }

    out1 = _ROOT / "configs" / "diffsbdd_real100.yaml"
    out2 = _ROOT / "configs" / "pocket2mol_real100.yaml"
    out3 = _ROOT / "configs" / "targetdiff_real100.yaml"
    write_yaml(out1, diff_cfg)
    write_yaml(out2, p2_cfg)
    td_cfg = {
        **base,
        "run_id": "targetdiff_real100",
        "model": {
            "type": "targetdiff",
            "n_samples": 20,
            "repo_root": "${env:TARGETDIFF_REPO}",
            "config_yaml": "${env:TARGETDIFF_SAMPLING_YAML}",
            "python_exe": "${env:TARGETDIFF_PYTHON}",
            "device": "cuda:0",
            "batch_size": 100,
            "sanitize": True,
            "extra_args": [],
        },
    }
    write_yaml(out3, td_cfg)
    print("Wrote", out1)
    print("Wrote", out2)
    print("Wrote", out3)
    if len(pocket_rows) < args.target:
        print(
            "NOTE: Fewer than target pockets — widen search, increase pagination, or pass --id-file.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
