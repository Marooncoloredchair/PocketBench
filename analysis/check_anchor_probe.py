#!/usr/bin/env python3
"""Check whether ``anchor_offset`` actually changes what Pocket2Mol sees.

The bridge shifts ``--center`` passed to ``sample_for_pdb.py`` while the on-disk
pocket PDB is unchanged. Pocket2Mol then *re-filters* protein atoms into a cubic
bbox around that center. If a 2 A offset keeps the same atoms in the box, the model
input is identical and ``anchor_offset`` is an inert knob (not evidence against the
first-atom story -- the probe never turned).

For each pocket under a benchmark generations tree, compare ``original`` vs
``anchor_offset_*`` condition folders:

  * pocket PDB byte identity (should match -- metadata-only perturbation)
  * bridge center / bbox and atom count inside the bbox (original vs shifted center)
  * sampled SMILES overlap (from ``*_smiles.txt`` or ``*_out.pt``)

Example::

    python analysis/check_anchor_probe.py \\
        --generations data/generations/run_pocket2mol_isr_smoke5

    python analysis/check_anchor_probe.py \\
        --generations data/generations/run_pocket2mol_isr_smoke5 \\
        --what-if-offsets 2 5 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ANCHOR_TAG_RE = re.compile(r"^anchor_offset_(?P<mag>[0-9]+(?:\.[0-9]+)?)$")


@dataclass
class BboxStats:
    center: list[float]
    bbox_edge: float
    n_atoms: int
    n_residues: int

    @property
    def half(self) -> float:
        return self.bbox_edge / 2.0


@dataclass
class ProbeRow:
    pocket_id: str
    anchor_tag: str
    offset_mag_a: float
    pdb_same_as_original: bool
    center_orig: list[float]
    center_shifted: list[float]
    bbox_edge: float
    n_atoms_orig: int
    n_atoms_shifted: int
    n_residues_orig: int
    n_residues_shifted: int
    atoms_delta: int
    n_smiles_orig: int
    n_smiles_anchor: int
    smiles_jaccard: float
    smiles_identical_order: bool
    verdict: str


def _read_pdb_atom_rows(pdb_path: Path) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
    """Heavy-atom coords and (chain, resseq, atom_name) keys from ATOM records."""
    coords: list[list[float]] = []
    keys: list[tuple[str, int, str]] = []
    with pdb_path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem and len(line) >= 16:
                elem = line[12:14].strip().upper()[:1]
            if elem in ("H", "D", "", "Q"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chain = line[21:22].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = 0
            atom_name = line[12:16].strip()
            coords.append([x, y, z])
            keys.append((chain, resseq, atom_name))
    if not coords:
        raise ValueError(f"No heavy ATOM records in {pdb_path}")
    return np.asarray(coords, dtype=np.float64), keys


def bridge_center_and_bbox(
    coords: np.ndarray,
    margin: float = 3.0,
    min_box: float = 23.0,
    max_box: float = 34.0,
) -> tuple[np.ndarray, float]:
    """Match ``pocket2mol_sample_drug_bridge._pocket_center_and_bbox``."""
    c = coords.mean(axis=0)
    extent = float(np.max(np.abs(coords - c)))
    bbox = float(max(min_box, min(max_box, 2.0 * extent + 2.0 * margin)))
    return c, bbox


def pca_offset_unit(coords: np.ndarray) -> np.ndarray:
    """First principal axis of pocket coords (same rule as anchor_offset perturbation)."""
    centered = coords - coords.mean(axis=0, keepdims=True)
    if centered.shape[0] >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        unit = vt[0]
    else:
        unit = np.array([1.0, 0.0, 0.0])
    n = float(np.linalg.norm(unit))
    return unit / n if n > 0 else np.array([1.0, 0.0, 0.0])


def parse_anchor_magnitude(tag: str) -> float | None:
    m = _ANCHOR_TAG_RE.match(tag)
    return float(m.group("mag")) if m else None


def atoms_in_bbox(
    coords: np.ndarray,
    keys: list[tuple[str, int, str]],
    center: np.ndarray,
    bbox_edge: float,
) -> BboxStats:
    half = bbox_edge / 2.0
    mask = np.max(np.abs(coords - center), axis=1) <= half
    idx = np.where(mask)[0]
    res_keys = {(keys[i][0], keys[i][1]) for i in idx}
    return BboxStats(
        center=center.tolist(),
        bbox_edge=bbox_edge,
        n_atoms=int(idx.size),
        n_residues=len(res_keys),
    )


def find_input_pdb(cond_dir: Path, pocket_id: str) -> Path | None:
    hits = sorted(cond_dir.glob(f"{pocket_id}_pocket2mol_in.pdb"))
    if hits:
        return hits[0]
    hits = sorted(cond_dir.glob("*_pocket2mol_in.pdb"))
    return hits[0] if hits else None


def find_smiles_paths(cond_dir: Path, pocket_id: str) -> list[Path]:
    paths: list[Path] = []
    for pat in (f"{pocket_id}_pocket2mol_out_smiles.txt", "*_pocket2mol_out_smiles.txt"):
        paths.extend(sorted(cond_dir.glob(pat)))
    # De-dupe while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_smiles_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    return lines


def load_smiles_from_condition(cond_dir: Path, pocket_id: str) -> list[str]:
    for p in find_smiles_paths(cond_dir, pocket_id):
        smi = load_smiles_lines(p)
        if smi:
            return smi
    pt_hits = sorted(cond_dir.glob(f"{pocket_id}_pocket2mol_out.pt")) or sorted(
        cond_dir.glob("*_pocket2mol_out.pt")
    )
    if not pt_hits:
        return []
    try:
        from sbdd_robust.models.pocket2mol_adapter import mols_from_pocket2mol_payload, load_pocket2mol_pt
        from rdkit import Chem

        raw = load_pocket2mol_pt(pt_hits[0])
        mols = mols_from_pocket2mol_payload(raw, sanitize=True)
        out: list[str] = []
        for m in mols:
            s = Chem.MolToSmiles(m) if m is not None else ""
            if s:
                out.append(s)
        return out
    except Exception:
        return []


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def classify_verdict(
    atoms_delta: int,
    smiles_jaccard: float,
    n_smiles_orig: int,
    n_smiles_anchor: int,
    smiles_identical_order: bool,
) -> str:
    if n_smiles_orig == 0 and n_smiles_anchor == 0:
        return "NO_OUTPUT"
    if atoms_delta != 0:
        return "PROBE_ACTIVE (bbox atom set changed)"
    if smiles_identical_order and smiles_jaccard == 1.0:
        return "PROBE_INERT (same bbox atoms, identical SMILES)"
    if smiles_jaccard >= 0.999:
        return "PROBE_INERT (same bbox atoms, same SMILES set)"
    return "PROBE_PARTIAL (same bbox atoms, different samples)"


def probe_pair(
    pocket_id: str,
    orig_dir: Path,
    anchor_dir: Path,
    anchor_tag: str,
) -> ProbeRow:
    mag = parse_anchor_magnitude(anchor_tag)
    if mag is None:
        raise ValueError(f"Not an anchor_offset tag: {anchor_tag}")

    pdb_o = find_input_pdb(orig_dir, pocket_id)
    pdb_a = find_input_pdb(anchor_dir, pocket_id)
    if pdb_o is None or pdb_a is None:
        raise FileNotFoundError(f"Missing input PDB under {orig_dir} or {anchor_dir}")

    pdb_same = pdb_o.read_bytes() == pdb_a.read_bytes()
    coords, keys = _read_pdb_atom_rows(pdb_o)
    center, bbox = bridge_center_and_bbox(coords)
    offset = pca_offset_unit(coords) * float(mag)
    shifted = center + offset

    st_o = atoms_in_bbox(coords, keys, center, bbox)
    st_a = atoms_in_bbox(coords, keys, shifted, bbox)

    smi_o = load_smiles_from_condition(orig_dir, pocket_id)
    smi_a = load_smiles_from_condition(anchor_dir, pocket_id)
    jac = jaccard(smi_o, smi_a)
    identical = smi_o == smi_a
    delta_atoms = st_a.n_atoms - st_o.n_atoms

    return ProbeRow(
        pocket_id=pocket_id,
        anchor_tag=anchor_tag,
        offset_mag_a=mag,
        pdb_same_as_original=pdb_same,
        center_orig=[round(x, 4) for x in st_o.center],
        center_shifted=[round(x, 4) for x in st_a.center],
        bbox_edge=round(st_o.bbox_edge, 3),
        n_atoms_orig=st_o.n_atoms,
        n_atoms_shifted=st_a.n_atoms,
        n_residues_orig=st_o.n_residues,
        n_residues_shifted=st_a.n_residues,
        atoms_delta=delta_atoms,
        n_smiles_orig=len(smi_o),
        n_smiles_anchor=len(smi_a),
        smiles_jaccard=round(jac, 4) if jac == jac else float("nan"),
        smiles_identical_order=identical,
        verdict=classify_verdict(delta_atoms, jac, len(smi_o), len(smi_a), identical),
    )


def print_what_if(coords: np.ndarray, keys: list[tuple[str, int, str]], offsets: list[float]) -> None:
    center, bbox = bridge_center_and_bbox(coords)
    print("\nWhat-if bbox atom counts (same PDB, PCA offset magnitude):")
    print(f"  bridge center={[round(x,3) for x in center.tolist()]}  bbox={bbox:.2f} A")
    unit = pca_offset_unit(coords)
    print(f"  PCA unit vector={[round(x,4) for x in unit.tolist()]}")
    base = atoms_in_bbox(coords, keys, center, bbox)
    print(f"  offset=0 A -> atoms={base.n_atoms} residues={base.n_residues}")
    for mag in offsets:
        st = atoms_in_bbox(coords, keys, center + unit * mag, bbox)
        print(
            f"  offset={mag:g} A -> atoms={st.n_atoms} residues={st.n_residues} "
            f"(delta {st.n_atoms - base.n_atoms:+d})"
        )


def discover_pockets(gen_root: Path) -> list[str]:
    return sorted(p.name for p in gen_root.iterdir() if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify anchor_offset changes Pocket2Mol's effective frame.")
    ap.add_argument(
        "--generations",
        required=True,
        type=Path,
        help="Benchmark generations root (e.g. data/generations/run_pocket2mol_isr_smoke5).",
    )
    ap.add_argument("--baseline-tag", default="original", help="Baseline condition folder name.")
    ap.add_argument(
        "--pockets",
        nargs="*",
        default=None,
        help="Restrict to these pocket IDs (default: all subdirs).",
    )
    ap.add_argument(
        "--what-if-offsets",
        nargs="*",
        type=float,
        default=None,
        help="Also print bbox atom counts for hypothetical offset magnitudes (A).",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    gen_root = args.generations.resolve()
    if not gen_root.is_dir():
        raise SystemExit(f"Not a directory: {gen_root}")

    pockets = args.pockets or discover_pockets(gen_root)
    rows: list[ProbeRow] = []

    print(f"Anchor probe - {gen_root}")
    print("=" * 72)

    for pid in pockets:
        pocket_root = gen_root / pid
        orig_dir = pocket_root / args.baseline_tag
        if not orig_dir.is_dir():
            print(f"\n[{pid}] SKIP — no {args.baseline_tag}/ folder")
            continue

        anchor_tags = sorted(
            d.name for d in pocket_root.iterdir() if d.is_dir() and d.name.startswith("anchor_offset_")
        )
        if not anchor_tags:
            print(f"\n[{pid}] SKIP — no anchor_offset_* folders")
            continue

        if args.what_if_offsets:
            pdb = find_input_pdb(orig_dir, pid)
            if pdb is not None:
                coords, keys = _read_pdb_atom_rows(pdb)
                print(f"\n[{pid}] {pdb.name}")
                print_what_if(coords, keys, args.what_if_offsets)

        for tag in anchor_tags:
            anchor_dir = pocket_root / tag
            try:
                row = probe_pair(pid, orig_dir, anchor_dir, tag)
            except (FileNotFoundError, ValueError) as exc:
                print(f"\n[{pid}/{tag}] ERROR: {exc}")
                continue
            rows.append(row)
            print(f"\n[{pid}] vs {tag} ({row.offset_mag_a:g} A along pocket PCA)")
            print(f"  pocket PDB unchanged vs original: {row.pdb_same_as_original}")
            print(f"  bridge center (orig):    {row.center_orig}")
            print(f"  bridge center (shifted): {row.center_shifted}")
            print(f"  bbox edge: {row.bbox_edge} A")
            print(
                f"  atoms in bbox: {row.n_atoms_orig} -> {row.n_atoms_shifted} "
                f"(delta {row.atoms_delta:+d}); residues {row.n_residues_orig} -> {row.n_residues_shifted}"
            )
            print(
                f"  SMILES: orig={row.n_smiles_orig} anchor={row.n_smiles_anchor} "
                f"jaccard={row.smiles_jaccard} identical_order={row.smiles_identical_order}"
            )
            print(f"  >>> {row.verdict}")

    if not rows:
        print("\nNo anchor_offset pairs probed.")
        return

    n_inert = sum(1 for r in rows if r.verdict.startswith("PROBE_INERT"))
    n_active = sum(1 for r in rows if r.verdict.startswith("PROBE_ACTIVE"))
    print("\n" + "=" * 72)
    print(f"Summary: {len(rows)} pair(s) - active={n_active} inert={n_inert}")
    if n_inert and not n_active:
        print(
            "Interpretation: anchor_offset did not change the atoms Pocket2Mol sees at these magnitudes.\n"
            "Try --what-if-offsets 5 10 15 or increase offset in the stress config before revising the hypothesis."
        )
    elif n_active:
        print(
            "Interpretation: center shift changes the bbox-filtered pocket — anchor_offset is a live knob.\n"
            "If metrics (QED/validity) still barely move, the model may be robust inside the changed frame."
        )

    if args.out_csv:
        import pandas as pd

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
