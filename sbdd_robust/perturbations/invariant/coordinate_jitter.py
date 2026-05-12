"""Add small isotropic Gaussian noise to coordinates (no chemistry change)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sbdd_robust.datasets.pocket import Pocket


def jitter_coordinates(
    pocket: Pocket,
    sigma: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Pocket:
    rng = rng or np.random.default_rng()
    p = pocket.copy()
    noise = rng.normal(scale=float(sigma), size=p.coords.shape).astype(np.float64)
    p.coords = pocket.coords + noise
    p.metadata = dict(pocket.metadata)
    p.metadata["perturbation_type"] = "invariant_coordinate_jitter"
    p.metadata["perturbation_tag"] = "coordinate_jitter"
    p.metadata["jitter_sigma"] = float(sigma)
    return p
