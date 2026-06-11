"""Population-level consistency filter: keep molecules only if featurization stress is small in summary space."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Tuple

from rdkit import Chem

MetricsFn = Callable[[Iterable[Chem.Mol]], dict[str, float]]


def population_consistency_filter(
    original_mols: List[Chem.Mol],
    perturbed_mols: List[Chem.Mol],
    metrics_fn: MetricsFn,
    *,
    qed_threshold: float = 0.10,
    sa_threshold: float = 0.10,
) -> tuple[bool, dict[str, Any]]:
    """
    Compare population mean QED and mean SA. Pass if both absolute deltas are below thresholds.

    ``metrics_fn`` must return dict with keys ``mean_qed`` and ``mean_sa`` (floats).
    """
    mo_orig = metrics_fn(original_mols)
    mo_pert = metrics_fn(perturbed_mols)
    dq = abs(float(mo_orig["mean_qed"]) - float(mo_pert["mean_qed"]))
    ds = abs(float(mo_orig["mean_sa"]) - float(mo_pert["mean_sa"]))
    passed = bool(dq < qed_threshold and ds < sa_threshold)
    info = {
        "original_mean_qed": mo_orig["mean_qed"],
        "perturbed_mean_qed": mo_pert["mean_qed"],
        "qed_delta": dq,
        "original_mean_sa": mo_orig["mean_sa"],
        "perturbed_mean_sa": mo_pert["mean_sa"],
        "sa_delta": ds,
        "passed_qed_filter": dq < qed_threshold,
        "passed_sa_filter": ds < sa_threshold,
        "passed": passed,
    }
    return passed, info


def consistency_filter_with_vina(
    original_mols: List[Chem.Mol],
    perturbed_mols: List[Chem.Mol],
    metrics_fn: MetricsFn,
    *,
    qed_threshold: float = 0.10,
    sa_threshold: float = 0.10,
    vina_threshold: float = 0.5,
    original_mean_vina: float | None = None,
    perturbed_mean_vina: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Same as ``population_consistency_filter`` plus optional mean Vina delta when scores exist.

    If ``original_mean_vina`` / ``perturbed_mean_vina`` are provided, ``|ΔVina|`` must be < vina_threshold.
    Otherwise Vina is skipped (passed_vina_filter True).
    """
    passed_base, info = population_consistency_filter(
        original_mols,
        perturbed_mols,
        metrics_fn,
        qed_threshold=qed_threshold,
        sa_threshold=sa_threshold,
    )
    if original_mean_vina is None or perturbed_mean_vina is None:
        info["original_mean_vina"] = original_mean_vina
        info["perturbed_mean_vina"] = perturbed_mean_vina
        info["vina_delta"] = None
        info["passed_vina_filter"] = True
        info["passed_combined"] = passed_base and info["passed_qed_filter"] and info["passed_sa_filter"]
        return bool(info["passed_combined"]), info

    dv = abs(float(original_mean_vina) - float(perturbed_mean_vina))
    info["original_mean_vina"] = float(original_mean_vina)
    info["perturbed_mean_vina"] = float(perturbed_mean_vina)
    info["vina_delta"] = dv
    info["passed_vina_filter"] = dv < vina_threshold
    info["passed_combined"] = (
        passed_base and info["passed_qed_filter"] and info["passed_sa_filter"] and info["passed_vina_filter"]
    )
    return bool(info["passed_combined"]), info


def originals_if_passed(
    original_mols: list[Chem.Mol],
    perturbed_mols: list[Chem.Mol],
    metrics_fn: MetricsFn,
    **kwargs: Any,
) -> tuple[list[Chem.Mol], dict[str, Any]]:
    """Return ``original_mols`` when the population filter passes; otherwise []."""
    ok, info = population_consistency_filter(original_mols, perturbed_mols, metrics_fn, **kwargs)
    return (list(original_mols) if ok else [], info)
