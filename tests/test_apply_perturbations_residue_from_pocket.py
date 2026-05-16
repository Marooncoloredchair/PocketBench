from __future__ import annotations

import pytest

from sbdd_robust.cli import _apply_perturbations
from tests.test_residue_mutation import _make_arg_like_pocket


def test_residue_id_from_pocket_requires_mutation_field():
    base = _make_arg_like_pocket()
    specs = [
        {
            "tag": "meaningful_ala",
            "type": "residue_mutation",
            "target_aa": "A",
            "residue_id_from_pocket": True,
        }
    ]
    with pytest.raises(ValueError, match="mutation_residue_id"):
        _apply_perturbations(base, specs, pocket_cfg={})


def test_residue_id_from_pocket_uses_stable_tag():
    base = _make_arg_like_pocket()
    specs = [
        {
            "tag": "meaningful_ala",
            "type": "residue_mutation",
            "target_aa": "A",
            "residue_id_from_pocket": True,
        }
    ]
    outs = _apply_perturbations(
        base, specs, pocket_cfg={"mutation_residue_id": "A:50"}
    )
    assert len(outs) == 1
    assert outs[0].metadata["perturbation_tag"] == "meaningful_ala"
    assert outs[0].metadata["perturbation_type"] == "meaningful"
