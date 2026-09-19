"""Validity, uniqueness, QED, and synthetic accessibility for generated molecules."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig

_SA_SCORER = None


def _load_sa_scorer():
    global _SA_SCORER
    if _SA_SCORER is not None:
        return _SA_SCORER
    path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
    if not os.path.isfile(path):
        _SA_SCORER = False
        return _SA_SCORER
    import importlib.util

    spec = importlib.util.spec_from_file_location("sascorer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sascorer"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _SA_SCORER = mod
    return _SA_SCORER


def _sa_score(mol: Chem.Mol) -> Optional[float]:
    s = _load_sa_scorer()
    if s is False:
        return None
    try:
        return float(s.calculateScore(mol))
    except Exception:
        return None


def is_valid_mol(mol: Chem.Mol) -> bool:
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        return True
    except Exception:
        return False


def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    if not is_valid_mol(mol):
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def summarize_molecules(mols: List[Chem.Mol]) -> Dict[str, Any]:
    n_total = len(mols)
    valid_flags = [is_valid_mol(m) for m in mols]
    n_valid = int(sum(valid_flags))
    validity = float(n_valid / n_total) if n_total else 0.0

    smiles_list = []
    for m, ok in zip(mols, valid_flags, strict=True):
        if not ok:
            continue
        smi = canonical_smiles(m)
        if smi is not None:
            smiles_list.append(smi)

    n_unique = len(set(smiles_list)) if smiles_list else 0
    uniqueness = float(n_unique / len(smiles_list)) if smiles_list else 0.0

    qeds: list[float] = []
    sas: list[float] = []
    mws: list[float] = []
    for m, ok in zip(mols, valid_flags, strict=True):
        if not ok:
            continue
        try:
            qeds.append(float(QED.qed(m)))
        except Exception:
            pass
        try:
            mws.append(float(Descriptors.MolWt(m)))
        except Exception:
            pass
        sa = _sa_score(m)
        if sa is not None:
            sas.append(sa)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity": validity,
        "n_unique_valid": n_unique,
        "uniqueness": uniqueness,
        "mean_qed": float(np.mean(qeds)) if qeds else float("nan"),
        "std_qed": float(np.std(qeds)) if qeds else float("nan"),
        "mean_sa": float(np.mean(sas)) if sas else float("nan"),
        "std_sa": float(np.std(sas)) if sas else float("nan"),
        "mean_mw": float(np.mean(mws)) if mws else float("nan"),
        "std_mw": float(np.std(mws)) if mws else float("nan"),
    }


def per_molecule_records(mols: List[Chem.Mol]) -> List[Dict[str, Any]]:
    """Per-molecule QED / molecular weight / SMILES for valid molecules.

    Persisting these prevents the SUBSTRATE_MISSING situation where per-molecule
    QED and MW are computed for the aggregate summary and then discarded, leaving
    downstream analyses (e.g. ΔQED~ΔMW) with no molecule-level substrate.
    """
    recs: List[Dict[str, Any]] = []
    for idx, m in enumerate(mols):
        ok = is_valid_mol(m)
        rec: Dict[str, Any] = {
            "mol_index": idx,
            "valid": bool(ok),
            "smiles": None,
            "qed": float("nan"),
            "mw": float("nan"),
            "sa": float("nan"),
        }
        if ok:
            rec["smiles"] = canonical_smiles(m)
            try:
                rec["qed"] = float(QED.qed(m))
            except Exception:
                pass
            try:
                rec["mw"] = float(Descriptors.MolWt(m))
            except Exception:
                pass
            sa = _sa_score(m)
            if sa is not None:
                rec["sa"] = float(sa)
        recs.append(rec)
    return recs
