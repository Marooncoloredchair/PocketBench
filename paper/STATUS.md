# PocketBench paper status (collaborator entry point)

**As of:** 2026-08-12  
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

Internal calibration note: `paper/monday_honest_summary.md`.

---

## Superseded — do not headline

| Claim | Why |
|-------|-----|
| Brittleness rate **1.0** as primary finding | Metric-scaling artefact; keep only with that explanation |
| Pocket2Mol coverage **13/47** as model-limited | Adapter input-format artefact (pocket-only PDBs vs full structure + ligand-centered bbox) |
| `pocket2mol_real47_coverage_v2.txt` old “**1/1**” line | Reporting error; run is **incomplete** (19/47 pockets) |

---

## In progress / incomplete

| Item | Status |
|------|--------|
| Pocket2Mol **corrected** real47 coverage | Partial: **19/47** pockets run; **16/19** valid on `original`; **no** corrected full-panel number yet |
| DiffSBDD real100 **SDF generations** | **Must regenerate for full panel:** local `data/generations/run_diffsbdd_real100` has SDFs for **10/99** pockets only; Drive `sbdd-robust-results/real100/` not mountable from this machine (sign-in required). Metrics CSV is in-repo (`data/results/metrics_per_condition__runreal100.csv`) |
| Cross-model ISR ranking (P2M / TargetDiff) | Preliminary; harness fixed on matched4; not panel-scale |
| Conditioning-faithfulness (PocketBench) | Not started in this repo |

---

## Canonical docs (after this file)

1. `paper/draft.md` — readable mirror of the corrected story  
2. `paper/biorxiv_submission/main.tex` — submission source  
3. `paper/nmi_cover_letter.md` — cover letter with corrected headline  
4. `paper/monday_honest_summary.md` — confidence table (solid vs preliminary)
