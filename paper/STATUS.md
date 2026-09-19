# PocketBench paper status (collaborator entry point)

**As of:** 2026-09-16  
**Repo:** PocketBench (`sbdd-robust`) — https://github.com/Marooncoloredchair/PocketBench  

Read this before `draft.md`, `biorxiv_submission/main.tex`, or any coverage / brittleness CSV.

---

## Trustworthy today (defend these)

| Claim | Evidence |
|-------|----------|
| DiffSBDD **robust to featurization noise** (shuffle + jitter) on **normalized** metrics | τ=0.10: **0/47** and **3/99** — `paper/normalized_brittleness.csv` |
| DiffSBDD **sensitive to pocket boundary** (`crop_radius_minus_1.5`) | median ΔQED **−0.060**, Wilcoxon **p=2.3×10⁻¹⁰** (n=99); replicated n=47 — `paper/crop_radius_wilcoxon.csv` |
| Old “brittleness rate **1.0**” is a **metric-scaling artefact** | raw counts + unnormalized SA in a [0,1]-tuned threshold; not the headline |
| DiffSBDD real47 metrics panel | `data/results/metrics_per_condition__run1778615375.csv` |
| Pocket2Mol **corrected** real47 panel (47 pockets, Unity Job B) | `data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv` (235 rows) |
| Pocket2Mol low v1 coverage (**13/47**) was largely an **adapter input-format artefact** | v2: **43/47** with `n_valid>0` on original; **30/34** of the gap closed by adapter fix — `paper/pocket2mol_real47_v2_analysis.csv` |

Internal calibration note: `paper/monday_honest_summary.md`.

---

## Pocket2Mol real47 v2 (corrected adapter, 2026-09-16)

Full panel merged from Unity array job `63114995` (47 shards → 235 rows).

| Metric | v1 (broken adapter) | v2 (fixed adapter) | DiffSBDD real47 (same analysis) |
|--------|---------------------|--------------------|---------------------------------|
| Original coverage (`n_valid>0`) | **13/47** (27.7%) | **43/47** (91.5%) | 47/47 |
| Mean validity on covered pockets | 1.000 | 1.000 | — |
| Normalized brittleness @ τ=0.10 | — | **13/47** (27.7%) | **0/47** |
| Crop ΔQED Wilcoxon (n pairs) | — | median **−0.006**, **p=0.78** (n=43) | median **−0.056**, **p=5.3×10⁻⁶** (n=47) |
| Crop ΔSA Wilcoxon | — | median **−0.46**, **p=1.8×10⁻⁴** (n=43) | **p=0.96** (n=47) |

**Adapter-fix attribution:** 30 of the 34 pockets that failed under v1 now generate (`88.2%` of the coverage gap). The remaining **4** zero-valid originals are **1DIZ, 2AV1, 2X4Q, 5T70**. Residual failures are likely true Pocket2Mol sampling limits, not the old PDB/bbox bug.

Canonical analysis output: `paper/pocket2mol_real47_v2_analysis.csv`  
Coverage prose: `paper/pocket2mol_real47_coverage_v2.txt`  
Reproduce: `python analysis/pocket2mol_real47_v2_analysis.py`

---

## Superseded — do not headline

| Claim | Why |
|-------|-----|
| Brittleness rate **1.0** as primary finding | Metric-scaling artefact; keep only with that explanation |
| Pocket2Mol coverage **13/47** as model-limited | Adapter input-format artefact; superseded by **43/47** v2 panel |
| `pocket2mol_real47_coverage_v2.txt` old “**1/1**” / “**19/47 incomplete**” lines | Reporting error; full 47-pocket panel now on disk |

---

## In progress / incomplete

| Item | Status |
|------|--------|
| Cross-model ISR ranking (P2M / TargetDiff) | Preliminary; harness fixed on matched4; not panel-scale |
| DiffSBDD real100 **SDF generations** | Local `data/generations/run_diffsbdd_real100` has SDFs for **10/99** pockets only |
| Conditioning-faithfulness (PocketBench) | Not started in this repo |
| Pocket2Mol v2 **4 residual zero-valid pockets** | Diagnose vs DiffSBDD on same pockets |
| **MW mechanism test** for the QED/SA dissociation | Run, but **panel-confounded** — see below |
| **DiffSBDD on real47 with archived molecules** | **Needed** to de-confound the MW test; config staged at `configs/experiments/diffsbdd_real47.yaml` (47 pockets, exact P2M panel match), not yet submitted |

### MW mechanism test (2026-09-16) — informative but confounded

QED and SA are both mass-dependent, so a single shared mass shift could masquerade as
two distinct failure modes. Tested directly; molecules re-parsed from generations
(`analysis/mw_mechanism_both_models.py` → `paper/mw_mechanism_both_models.csv`).

| Model | Panel | n | median ΔMW (crop −1.5 Å) | p | median ΔQED | median ΔSA |
|-------|-------|---|--------------------------|---|-------------|------------|
| Pocket2Mol | real47 | 43 | **−23.04 Da** (lighter) | 3.2×10⁻⁴ | −0.006 (ns) | **−0.46** |
| DiffSBDD | real100_vina | 99 | **+8.95 Da** (heavier) | 0.011 | **−0.054** | +0.109 |

Directions are **opposite** and both significant, which argues *against* "one mass shift
scored two ways." MW→metric coupling is also model-specific: for DiffSBDD, ΔMW explains
**72.5%** of ΔSA (Pearson +0.851) and **31.3%** of ΔQED (−0.559); for Pocket2Mol it explains
**11%** of ΔQED and **2.4%** of ΔSA (ns). Featurization control is clean — no significant
ΔMW under `atom_shuffle` or `coordinate_jitter` for either model (all p>0.09), so the mass
shift is boundary-specific.

> **Do not headline this as architectural.** The two panels are **disjoint (0 shared
> pockets)**; DiffSBDD's real47 molecules were never archived off Colab. Model effect and
> pocket-set effect are **not separable** until DiffSBDD is re-run on real47 with molecules
> retained.

---

## Canonical docs (after this file)

1. `paper/draft.md` — readable mirror of the corrected story  
2. `paper/biorxiv_submission/main.tex` — submission source  
3. `paper/nmi_cover_letter.md` — cover letter with corrected headline  
4. `paper/monday_honest_summary.md` — confidence table (solid vs preliminary)
