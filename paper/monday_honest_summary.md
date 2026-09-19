# Monday honest summary — what to trust vs what is still in flux

**Audience:** internal sync with Arne's team (and anyone else who needs calibrated confidence, not headline numbers).  
**Date:** 2026-06-15  
**Author:** Terrell Osborne

This note separates **solid, panel-scale DiffSBDD findings** (the paper's core claim) from **preliminary cross-model ISR work** where harness bugs and incomplete perturbation coverage still dominate interpretation.

---

## TL;DR for the meeting

| Claim | Status | One-line confidence |
|-------|--------|---------------------|
| DiffSBDD is robust to featurization noise (shuffle + jitter) on normalized metrics | **SOLID** | Replicated on n=47 and n=99; ≤3% brittleness at τ=0.10 |
| DiffSBDD is sensitive to pocket boundary (crop −1.5 Å) on QED | **SOLID** | Large paired Wilcoxon effect on both panels; ~54% of pockets lose >0.05 QED |
| Pocket2Mol boundary sensitivity vs DiffSBDD (ISR ranking) | **PRELIMINARY — balanced harness** | Selective ghost masking v3: **4/4** both frame tags; P2M ISR **1.21** vs DiffSBDD **2.34** on matched4 |
| TargetDiff shows high ISR on 4-pocket smoke panel | **PRELIMINARY** | Rerun in progress; prior 1I4F face_peel empty |
| Cross-model ISR ordering | **NOT YET TRUSTWORTHY** | DiffSBDD local face_peel + TargetDiff reruns still running; n=4 |

---

## SOLID findings (DiffSBDD, 47- and 99-pocket panels)

These results use the same perturbation protocol, normalized chemistry metrics, and archived Colab/local CSVs referenced in `paper/draft.md`. I would defend these in a methods discussion or reviewer response **today**.

### 1. Featurization noise → negligible brittleness

- **Panels:** real47 (n=47) and real100 (n=99) RCSB complexes.
- **Perturbations:** `atom_shuffle`, `coordinate_jitter` (+ crop variants, analyzed separately).
- **Metric:** normalized brittleness on validity, uniqueness, mean QED, std QED (`paper/normalized_brittleness.csv`).
- **Result:** at τ=0.10, brittleness is **0/47 (0%)** and **3/99 (3.0%)**.
- **Artifacts:** `data/results/metrics_per_condition__run1778615375.csv` (47-pocket Colab run); real100 aggregates in `paper/pocket_boundary_sensitivity_summary.csv`.

### 2. Pocket boundary contraction → systematic QED degradation

- **Perturbation:** `crop_radius_minus_1.5` vs `original`.
- **Wilcoxon paired ΔQED** (`paper/crop_radius_wilcoxon.csv`):
  - real47: W=159, **p=5.3×10⁻⁶**, rank-biserial **r≈−0.72**, median ΔQED **−0.056**
  - real100: W=658, **p=2.3×10⁻¹⁰**, **r≈−0.73**, median ΔQED **−0.060**
- **Fraction losing >0.05 QED:** ~51% (47-panel) / ~54% (99-panel) per `paper/crop_radius_effect.csv`.
- **SA (synthetic accessibility):** no significant paired shift on either panel (p≈0.96 / 0.22) — sensitivity is **QED-specific**, not global chemistry collapse.

### 3. Pocket Boundary Sensitivity Index (PBSI / ISR framing)

- **Dose-response slope** (median |dQED/dΔr|): **0.018** (47) / **0.023 Å⁻¹** (99).
- **Boundary swing exceeds featurization-noise floor:** **74.5%** (47) / **77.8%** (99) of pockets.
- **Median SNR (1.5 Å crop vs shuffle+jitter std):** **1.7×** (47) / **2.6×** (99).
- Source: `paper/pocket_boundary_sensitivity_summary.csv`.

### 4. Initialization Sensitivity Ratio (ISR) — DiffSBDD only

ISR = median |ΔQED| under frame perturbations / median |ΔQED| under featurization noise.

| Panel | n pockets | Frame tags available | ISR (mean QED) | 95% CI |
|-------|-----------|----------------------|----------------|--------|
| real47 Colab | 47 | `crop_radius_minus_1.5` only | **1.98** | [1.42, 2.54] |
| real100 | 99 | `crop_radius_minus_1.5` only | (same protocol; PBSI tables above) | — |
| matched4 slice | 4 | `crop_radius_minus_1.5` only (no face_peel in Colab export) | **2.34** | [1.56, 3.69] |

**Caveat (minor):** Colab DiffSBDD stress export does **not** include `face_peel_0.25`; ISR uses crop only. That is fine for the **within-DiffSBDD** boundary-vs-noise contrast but limits apples-to-apples comparison with Pocket2Mol/TargetDiff matched panels that include face_peel.

### 5. What I would say out loud

> "On two independent panels totaling ~150 unique targets, DiffSBDD is stable to representational noise but systematically moves drug-likeness when we change pocket cropping by 1.5 Å. That replicates in direction, magnitude, and significance. The reliability problem we can document today is **pocket definition**, not atom order or sub-ångström jitter."

---

## PRELIMINARY / IN-FLUX findings (cross-model ISR and Pocket2Mol)

**Do not** present the following as established comparative conclusions until the items in §"Open debugging" are closed.

### Matched 4-pocket ISR panel (1AO7, 1B0R, 1HXC, 1I4F)

Fixed pocket set, five conditions each: `original`, `atom_shuffle`, `coordinate_jitter`, `crop_radius_minus_1.5`, `face_peel_0.25`.  
Source: `paper/matched_isr_4pocket.csv`.

**Pre-fix (ghost atoms):** Pocket2Mol ISR = **0.0** — frame QED identical to original on 3/4 pockets.

**Post-fix v2 (aggressive bbox mask):** stripped *all* non-pocket atoms inside the box → 0 mols on most frame tags; inflated ISR **5.91** from asymmetric coverage.

**Post-fix v3 (selective ghost-atom mask):** `metrics_per_condition__runpocket2mol_isr_matched4_v3.csv` — only removes atoms present in the **reference** pocket tensor but absent from the **perturbed** tensor (see `paper/pocket2mol_frame_failure_diag.csv`).

| Model | ISR | Frame median \|ΔQED\| | Feat median \|ΔQED\| | Coverage notes |
|-------|-----|------------------------|-------------------------|----------------|
| DiffSBDD | **2.34** [1.56, 3.69] | 0.069 | 0.029 | 4/4; crop only until local face_peel run completes |
| Pocket2Mol (v3) | **1.21** [0.18, 2.33] | **0.038** | 0.031 | **4/4 both frame tags** |
| TargetDiff | **9.83** [0.75, 14.2] | 0.048 | 0.005 | prior smoke5; matched4 rerun queued |

**Per-pocket frame QED movement (v3, all valid):**

| Pocket | crop −1.5 Å | face_peel 0.25 |
|--------|-------------|----------------|
| 1AO7 | 0.606 → **0.549** (Δ=0.057) | 0.606 → **0.593** (Δ=0.013) |
| 1B0R | 0.693 → **0.597** (Δ=0.096) | 0.693 → **0.656** (Δ=0.037) |
| 1HXC | 0.645 → **0.651** (Δ=0.006) | 0.645 → **0.657** (Δ=0.012) |
| 1I4F | 0.580 → **0.463** (Δ=0.117) | 0.580 → **0.619** (Δ=0.039) |

**Interpretation:** With a trustworthy harness, Pocket2Mol **does** respond to frame perturbations on all four pockets, but its matched4 ISR is **below** DiffSBDD's crop-only ISR (~1.2 vs ~2.3). The earlier "P2M ≫ DiffSBDD" story was an artefact of aggressive masking + uneven coverage. Cross-model ranking remains preliminary until DiffSBDD face_peel and TargetDiff matched4 reruns land.

**Why Pocket2Mol ISR was 0.0 (not "Pocket2Mol is invariant"):**

1. **Input-format bug (fixed):** Bridge originally fed **pocket-only cropped PDBs** with pocket-centroid anchoring. Pocket2Mol expects **full protein PDB** + **ligand centroid** + cubic bbox (`sample_for_pdb.py` workflow). That caused **0 SMILES** on many pockets (e.g. 1B0R, 1I4F) and **13/47 (28%)** coverage on the real47 baseline (`metrics_per_condition__runpocket2mol_real47_full.csv`).

2. **Ghost-atom bug (fixed v3):** Aggressive masking removed all non-pocket atoms in the bbox (see diag: 37 vs 378 atoms for 1AO7 crop). **Selective masking** removes only `reference_pocket_keys − perturbed_pocket_keys` inside the box (`sbdd_robust/datasets/pdb_merge.py`, `pocket2mol_ghosts.py`).

3. **Coverage recheck (10 failed pockets):** With the full-PDB adapter, **8/10** previously failed pockets now generate valid molecules (`paper/pocket2mol_coverage_recheck_summary.txt`). Extrapolating → **~80% of the original 34/47 failures were harness/input-format**, not intrinsic Pocket2Mol inability.

### Pocket2Mol coverage is largely input-format dependent

| Metric | Before adapter fix | After fix (evidence) |
|--------|-------------------|----------------------|
| real47 `original` validity>0 | **13/47 (28%)** | Recheck **8/10** recovered; matched4 **4/4** with valid QED |
| Likely true hard failures | unknown | **~7/47** extrapolated + **1DIZ, 2AV1** still fail recheck |

**Honest wording:** "Pocket2Mol's low coverage on our first pass was mostly our fault — wrong PDB scope and center. True model-limited pockets exist but are a minority we have not fully enumerated."

### TargetDiff smoke (5-pocket / matched4)

- Runs on cropped pocket tensors (different architecture path than Pocket2Mol full-PDB merge).
- High ISR on 4-pocket panel is **suggestive** but **n=3–4** with one missing face_peel row.
- Sanity variants in `paper/matched_isr_sanity.csv` show ISR is sensitive to excluding 1I4F — treat as **exploratory**.

### Anchor-offset smoke (Pocket2Mol only)

`data/results/isr_anchor510_compare.csv`: ISR **3.2–4.8** on 4-pocket anchor-offset + crop + face_peel panel — but this predates ghost-atom masking fix and mixed perturbation types. **Do not cite** until reconciled with matched4 v2.

---

## Open debugging (as of this writing)

| Item | Status |
|------|--------|
| Enable selective ghost masking (reference − perturbed atom keys) | **Done** |
| Re-run matched4 P2M (`pocket2mol_isr_matched4_v3`) | **Done** — 4/4 both frame tags; ISR **1.21** |
| DiffSBDD matched4 + face_peel locally | **Running** (Biopython import patched in `DiffSBDD/lightning_modules.py`) |
| TargetDiff matched4 rerun | **Queued** (sequential batch after DiffSBDD) |
| P2M matched10 + real47 v2 re-benchmark | **Queued** in same batch |
| Update `paper/matched_isr_4pocket.csv` | **Done** (v3 P2M row) |

---

## Recommended talking points for Arne's team

### Say with confidence

1. **Pocket boundary definition dominates featurization noise for DiffSBDD** — replicated, significant, effect-sized.
2. **Normalized brittleness ≤3%** under shuffle+jitter; the old "brittleness=1.0 everywhere" headline was a metric-scaling artefact.
3. **Our benchmark protocol and archived CSVs are real**; the core single-model story does not depend on Pocket2Mol or TargetDiff.

### Flag as preliminary

1. **Any cross-model ISR ranking** — harness asymmetry (full PDB vs pocket tensor, center definition, bbox masking) swamps biology until aligned.
2. **Pocket2Mol coverage numbers before June 2026** — treat as **invalid**; ~80% of failures were adapter format, not model limits.
3. **TargetDiff ISR ~10 on 4 pockets** — interesting hypothesis, not a result.

### What we tested (2026-06-15)

> Re-ran the 4-pocket Pocket2Mol panel with bbox masking on **both** crop and face_peel (no ghost-atom fallback). **ISR moved off 0.0 to 5.91** [2.13, 6.87] with frame median |ΔQED| = 0.186. Generation under masking is **asymmetric** (1AO7 face_peel only; 1B0R crop only; 1HXC/1I4F neither). Next step: diagnose sparse-pocket failures vs tune masking — still n=4 for cross-model claims.

---

## Key file index

| File | Role |
|------|------|
| `paper/draft.md` | Full manuscript narrative (DiffSBDD-focused) |
| `paper/normalized_brittleness.csv` | Solid featurization-noise brittleness |
| `paper/crop_radius_wilcoxon.csv` | Solid paired boundary tests |
| `paper/pocket_boundary_sensitivity_summary.csv` | Solid PBSI / SNR aggregates |
| `data/results/metrics_per_condition__run1778615375.csv` | DiffSBDD 47-pocket Colab metrics |
| `data/results/metrics_per_condition__runpocket2mol_isr_matched4_v2.csv` | Post-fix P2M matched4 metrics (bbox masking) |
| `paper/matched_isr_4pocket.csv` | Cross-model ISR panel (includes v2 P2M row) |
| `paper/pocket2mol_coverage_recheck.csv` | P2M harness recovery audit |
| `configs/experiments/pocket2mol_isr_matched4.yaml` | Matched4 P2M panel config |
| `sbdd_robust/models/pocket2mol_adapter.py` | Full-PDB merge + bbox masking logic |

---

## Revision log

- **2026-06-15:** Selective ghost masking v3; matched4 **4/4** frame coverage; P2M ISR **1.21**; sequential batch started for DiffSBDD/TargetDiff/matched10/real47.
