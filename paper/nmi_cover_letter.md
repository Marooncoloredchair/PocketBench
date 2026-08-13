# Cover letter (Nature Machine Intelligence)

Dear Editors,

We submit “Reliability of generative structure-based drug design models under pocket featurization stress” for consideration as a **Research Article** in *Nature Machine Intelligence*.

Structure-based drug design (SBDD) generative models are usually scored on static pockets, which tells us **what** molecules they emit but not whether those outputs are **reliable** when the pocket encoding changes—atom order, sub-angstrom noise, and cropping choices that workflows routinely apply. We introduce **the first systematic perturbation-based reliability benchmark** for SBDD generative models and instantiate it as the open-source pipeline **PocketBench** (https://github.com/Marooncoloredchair/PocketBench). On **two independent panels** (n = 47 and n = 99 RCSB complexes) we evaluate DiffSBDD under chemistry-preserving stresses plus a 20-pocket meaningful-residue mutation screen with optional rescoring; Pocket2Mol is included with an explicit caveat on coverage measurement (below).

**Headline finding.** DiffSBDD is **robust to featurization noise** on normalized chemistry metrics (validity, uniqueness, mean/std QED): normalized brittleness is **0/47** and **3/99** at τ = 0.10. The same model is **systematically sensitive to pocket boundary definition**: `crop_radius_minus_1.5` yields median ΔQED **−0.060** (Wilcoxon p = **2.3×10⁻¹⁰**, rank-biserial r ≈ −0.73) on the n = 99 panel, replicated on n = 47. An earlier “brittleness rate 1.0” claim is a **metric-scaling artefact** (raw counts and unnormalized SA in a [0,1]-tuned threshold) and is not our headline. For Pocket2Mol, an original **13/47** coverage figure was largely an **adapter input-format artefact** (pocket-only PDBs vs full structure + ligand-centered bbox); a corrected re-benchmark is in progress and **no corrected coverage number is claimed**.

This work matters because leaderboard validity alone can hide **input-side failure modes**—especially arbitrary pocket-definition policy. Our benchmark couples coverage, normalized brittleness, and pocket-boundary sensitivity so that SBDD reliability can be reported alongside conventional chemistry metrics.

A **ChemRxiv** preprint is available (doi:10.26434/chemrxiv.15003499). Code and frozen metrics are in the repository; larger generation archives are referenced in the manuscript data statement. See also `paper/STATUS.md` for what is trustworthy vs superseded as of the latest revision.

**Suggested reviewers (no conflicts):**

1. **Charlotte M. Deane** — University of Oxford, UK (computational structural biology; method evaluation ecosystems including pose and interaction checks relevant to pocket conditioning).  
2. **Andreas Bender** — University of Cambridge, UK (computational medicinal chemistry; generative molecular design evaluation).  
3. **Benoit Baillif** — University of Cambridge, UK (GenBench3D and structure-based 3D generative model benchmarking).

We appreciate your consideration.

Sincerely,  
Terrell Osborne (on behalf of all authors)

---

*Word count: ~420.*
