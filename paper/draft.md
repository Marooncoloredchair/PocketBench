# Reliability of generative structure-based drug design models under pocket featurization stress

**Author:** Terrell Osborne¹, **Affiliation:** ¹Department of Computer Science, University of Rhode Island, Kingston, RI 02881, USA. **Correspondence:** terrell_osborne@uri.edu

**Submission manuscript:** the canonical sources for bioRxiv / journal submission are `paper/biorxiv_submission/main.tex` + `paper/biorxiv_submission/references.bib`; the compiled PDF is `paper/biorxiv_submission/main.pdf` (regenerate with your LaTeX toolchain or Tectonic). This `draft.md` mirrors that content for quick reading—figure numbers below match the current TeX float order.

---

## Abstract

Generative models for structure-based drug design (SBDD) are evaluated almost exclusively on molecular output quality — validity, uniqueness, and synthetic accessibility computed from a single fixed featurization of the target binding pocket. This design measures what models can generate but not whether their outputs are reliable under the input variation that drug discovery workflows routinely introduce. Here we introduce a perturbation-based reliability benchmark and apply it to two leading SBDD generative models, DiffSBDD and Pocket2Mol, across 47 diverse protein–ligand complexes from the RCSB Protein Data Bank. Four featurization stresses — atom-order shuffle, sub-ångström coordinate jitter, and ±1.5 Å crop-radius adjustment — aim to preserve binding-site chemistry while varying how the pocket tensor is constructed from the same structural model. DiffSBDD achieves full generation coverage across all 47 pockets but is flagged brittle on every pocket at threshold τ ≤ 0.10, with brittleness rates of 0.979 and 0.915 at τ = 0.15 and 0.20 respectively. Pocket2Mol generates valid molecules on only 13 of 47 pockets and exhibits brittleness rate 1.0 on those covered cases. Individual failures are substantial: coordinate jitter alone reduces molecule validity for PDB entry 3RFM from 0.85 to 0.50, a collapse attributable to perturbations smaller than typical crystallographic uncertainty. Complementary nearest-residue mutation experiments on 20 pockets reveal a second failure mode: DiffSBDD responds to only 32 of 40 biologically meaningful perturbations, with directionally interpretable responses in 65.6% of responsive cases and inconsistent Vina docking score shifts under tryptophan substitution. Together these results demonstrate a dual reliability failure in current SBDD generative models — fragility to representational noise that should be inconsequential, and inconsistent sensitivity to biological signals that should be decisive. We argue that SBDD benchmarks should routinely report generation coverage, brittleness under featurization stress, and meaningful-perturbation responsiveness alongside conventional quality metrics.

---

## Introduction

Generative structure-based drug design seeks ligands that complement a resolved or predicted binding site [1]. Contemporary pipelines couple geometric deep learning with molecular sampling and are benchmarked almost exclusively on static pocket coordinates: researchers report distributions of validity, synthesizability, drug-likeness, and downstream proxy tasks on held-out targets [2]. Those evaluations answer what molecules a model proposes on a given representation; they do not, without further tests, answer whether those proposals are *reliable* when the ostensibly same pocket is re-encoded through orderings, noise, or cropping policies that leave chemical identity nominally unchanged [3].

The benchmark we present is open source and reproducible: code, configuration, archived metrics, and figure-generation scripts are packaged together so independent groups can rerun the same perturbation protocol and comparative analyses.

Reliability matters for drug discovery because structural biology workflows routinely expose models to heterogeneous preprocessing choices: alternate atom orderings in PDB-derived tensors, coordinate noise within experimental uncertainty, and geometric cropping heuristics that define “the pocket.” If benchmark scores swing under such featurization-preserving variants, reported performance conflates molecular competence with sensitivity to arbitrary pocket definition. The methodological gap is that mainstream SBDD benchmark suites emphasize cross-target generalization and scalar quality metrics on **static** pocket encodings, not stress tests that deliberately re-encode the ostensibly same site through ordering noise, coordinate jitter, or pocket-definition heuristics.

We address this gap with a controlled protocol. We apply four pocket featurization stresses, regenerate molecules per condition, and flag brittleness when any monitored metric's standard deviation across those conditions exceeds a threshold. We instantiate the protocol on a forty-seven-complex panel with DiffSBDD and Pocket2Mol, resolve which summary columns dominate the brittle flag pocket-by-pocket (Fig. 1), report cross-model coverage and brittleness on covered pockets (Fig. 2), document condition-level validity collapse and pooled synthetic-accessibility drift (Fig. 3, Fig. 4), quantify threshold sensitivity for DiffSBDD (Fig. 5, Table 1), and add a twenty-pocket **meaningful mutation** benchmark with optional Vina rescoring (Fig. 6).

---

## Results

### Brittleness under pocket featurization stress

We evaluated DiffSBDD and Pocket2Mol on forty-seven pocket–model pairs using one reference pocket and four featurization stress conditions per target (Methods). Two of those conditions adjust crop radius around a fixed ligand anchor; they **change which heavy atoms enter the pocket tensor** and are therefore not rigid invariances of a single labelled atom list, even though they are applied to the same crystallographic entry—the same caveat applies to every crop-related result below.

*Generation coverage* is the fraction of pockets for which the model produced at least one valid molecule on the **original** (unperturbed) pocket. DiffSBDD achieved full coverage (**47/47**). Pocket2Mol achieved partial coverage (**13/47**); thirty-four pockets yielded zero valid molecules on the reference structure and are **generation failures** on the static conditioning signal, not brittleness under featurization stress.

*Brittleness rate* is computed **only on covered pockets** (original-pocket *n*valid > 0). For DiffSBDD the covered set is the full panel. At **τ ≤ 0.10**, every covered pocket was brittle (**brittleness rate 1.0**; **47/47**). For Pocket2Mol, recomputation on the **thirteen** covered pockets yields **brittleness rate 1.0** at the thresholds reported in the comparative export (Fig. 2). Headline brittle counts that pool all forty-seven targets without conditioning on coverage conflate unconditional failure with instability under featurization stress; we stratify all brittleness statistics accordingly.

At τ = 0.10, the brittle flag is explained by different summary columns on different pockets: we assign each covered pocket a **dominant driver**, defined as the metric with the largest empirical standard deviation across the four stress conditions among all table columns whose standard deviation exceeds τ (Methods). On DiffSBDD, **unique valid-molecule count** dominates for twenty-six pockets, **mean SA** for seventeen, **std SA** for three, and **batch size** *n*total for one (Fig. 1), so the headline rate is not driven by a single noisy scalar.

Condition-level analysis exposes large excursions under coordinate and tensor noise that nominally preserve chemistry. Coordinate jitter reduced validity for PDB entry **3RFM** from **0.85** on the original pocket to **0.50** on the jittered pocket under matched sampling (Fig. 3). Across both models, **mean synthetic accessibility (SA)** shifted **upward** when metrics were aggregated across pockets and stress conditions versus the original-pocket baseline, indicating systematic synthesizability drift under input variation rather than isolated outliers (Fig. 4).

### Threshold sensitivity analysis

Brittleness classifications depend on the dispersion threshold τ. For DiffSBDD on the forty-seven covered pockets, the brittleness rate remained **1.0** at **τ = 0.05** and **τ = 0.10**, then **relaxed** to **0.979** (**46/47**) at **τ = 0.15** and **0.915** (**43/47**) at **τ = 0.20** (Fig. 5; Table 1). The plateau at τ ≤ 0.10 shows near-universal brittle dispersion on conservative thresholds; departures appear only at lenient τ.

### Meaningful side-chain perturbations (DiffSBDD, twenty-pocket panel)

We stress-tested DiffSBDD on **20 pockets** with **nearest-residue** **alanine** vs **tryptophan** substitutions that preserve the backbone but change local chemistry; the mutated residue is chosen by proximity to the reference ligand centroid (Methods). Tags `meaningful_ala` and `meaningful_trp` are **fixed across pockets** so summaries remain comparable despite different absolute residue IDs.

Pooling **40 pocket–mutation pairs** (20 × 2), we classified a pair as **responsive** if **|Δ| > 0.10** versus the **original** pocket in **at least one** of: validity, mean QED, mean SA, or mean **AutoDock Vina** score (20 Å box, pocket heavy-atom receptor; Methods). **Thirty-two pairs (80%)** were responsive; **eight** were not. Among responsive pairs, **65.6%** (21/32) passed a **QED–validity directionality** screen (no strong opposing shifts; see `analysis/meaningful_perturbation_analysis.py`), and **59.4%** (19/32) passed a coarse **Vina–QED** concordance check. **Alanine** mutations **predominantly shift mean Vina toward less negative kcal/mol** (predicted affinity **weakens**; positive Δ when defined as mutant − original), consistent with **trimming sidechain interaction surface**. **Tryptophan** responses were **variable** across pockets. **Three of twenty pockets** showed **no** responsive pair for **either** mutation—suggesting binding contexts **dominated by backbone or contact patterns** **insensitive** to identity at the nearest mutated sidechain.

Summary tables and a delta heatmap are in `paper/meaningful_perturbation_summary.csv` and Fig. 6 / `paper/figures/meaningful_deltas_heatmap.pdf`, updated after docking rescoring on archived SDFs (`run_diffsbdd_meaningful_real20`).

### Docking score sensitivity under meaningful perturbations

Vina **mean** scores (kcal/mol; **more negative** = more favorable) illustrate **pocket-specific** sensitivity. Representative **shifts** (mutant − original) include: **1B0R** original **−3.824** → alanine **−3.496** (**Δ = +0.328**); **5U98** **−4.045** → alanine **−3.541** (**Δ = +0.504**); **3UPR** **−3.913** → tryptophan **−4.255** (**Δ = −0.342**, **stronger** predicted binding in this case). These examples sit alongside the aggregate pattern (Ala tending toward **less negative** means; Trp heterogeneous) and underscore that **meaningful perturbations induce interpretable order-of-magnitude score motion** on par with property-metric shifts.

**Summary of reliability failure modes.** Taken together, **DiffSBDD exhibits a dual reliability failure**: **(i)** **brittleness** to **benign-seeming pocket featurization noise** (atom order, sub-angstrom jitter, crop) that discovery workflows should largely collapse to a single binding hypothesis, and **(ii)** **uneven sensitivity** to **biologically meaningful** pocket edits that should yield **coherent**, rankable changes across chemistry and docking summaries. Strong static-pocket benchmarks do not resolve which failure dominates on a given target; both must be reported.

### Brittleness is stable across sample counts on a subset

The main panel fixes **n = 20** samples per pocket and condition. On the ten pockets with highest reference-pocket validity from the archived DiffSBDD export, we repeated the four stress conditions with **n = 50** while holding extraction radius and perturbation tags fixed. At **τ ∈ {0.05, 0.10}**, **brittleness rates stayed 1.0** (**10/10** brittle pocket–model pairs) for **both** budgets (`paper/nsamples_sensitivity.csv`); the check is annotated on **Fig. 5** rather than as a separate figure of identical 1.0 bars.

---

## Discussion

Pocket featurization stresses make input-side fragility measurable alongside conventional chemistry metrics. The combination of **near-saturated DiffSBDD brittleness at conservative τ**, **full reference-pocket coverage**, and **dramatic validity collapse on individual pockets** (for example 3RFM) demonstrates that strong static-pocket scores do not imply stable behaviour under featurization-preserving stress. The **meaningful mutation** panel adds a complementary lens: many pocket–mutation pairs produce large summary shifts, yet **three pockets** are mute to **both** Ala and Trp, and concordance between **cheap properties** and **Vina** is imperfect—consistent with the **dual reliability failure** summarized in Results (**featurization noise** vs **biological signal**). Pocket2Mol on the same panel illustrates a **distinct baseline problem**: incomplete coverage on the original pocket and high brittleness among covered pockets emphasize that models can fail before perturbation is applied; reporting coverage is therefore mandatory when interpreting brittle counts.

These findings carry implications for **benchmark design**. Leaderboards should require (i) explicit generation coverage on the reference pocket, (ii) brittleness statistics computed on covered targets, (iii) tabulated threshold sensitivity rather than single-τ headlines, and (iv) condition-level traces for validity and synthesizability where policy decisions depend on absolute thresholds. Review criteria for generative SBDD manuscripts should treat input-variation stress tests as first-class artefacts alongside validity histograms.

### Mechanistic hypotheses for observed brittleness

SE(3)-equivariant diffusion models such as DiffSBDD enforce consistent predictions under global Euclidean rigid motions of the input point cloud; that constraint does not extend by construction to the featurization stresses exercised here. Changes in crop radius **change the input atom set** included in the pocket tensor, so successive conditions need not be related by any single rigid transform of a fixed labelled set. Coordinate jitter leaves nominal chemistry intact but can **move atom pairs across distance cutoffs** used to build *k*-nearest-neighbour or radius graphs or analogous edge masks, producing **discrete edits** to pocket graph topology even for small displacements. Atom-order shuffling preserves Cartesian coordinates but can **couple to any featurization or batching path** that is not permutation-symmetric at every stage (for example ordering-dependent encodings or pooling). Under these mechanisms, metric dispersion across conditions is compatible with equivariant message passing on each fixed graph while the **graphs themselves** and **tensor layouts** differ across conditions.

Pocket2Mol’s autoregressive construction conditions each extension step on the current partial ligand and on pocket context processed through graph neural network modules trained on crystallographic pockets drawn from a finite data distribution. Reference-pocket **coverage failure** (many targets with zero valid molecules before perturbation) is therefore plausibly linked to **out-of-distribution pocket geometry or featurization** relative to that training regime, including crop-induced sparsity and jitter-induced edge changes, rather than to small oscillations around a single high-quality decoding trajectory. Where generation does start, sequential sampling can **amplify** small contextual shifts into contrasting completion paths; distinguishing **decode-time instability** from **hard refusal to initiate** requires the coverage stratification already adopted in the Results.

These accounts are **mechanistic hypotheses** framed to connect observed coverage and brittleness patterns to known architectural and featurization commitments. They are not established causes in the present benchmark: **confirming** which of the above channels dominates for each model—and whether auxiliary factors such as stochastic decoding, numerical precision, or training-set bias contribute—will require **targeted ablations** (for example fixed graph extraction, permutation-stress tests of encoders, and controlled crop or noise sweeps). Pinpointing the **dominant locus of brittleness within each architecture** remains an essential direction for future work.

**Limitations.** Brittleness is defined through a **threshold τ** on empirical standard deviations of summary metrics across four featurization-stress conditions; τ trades sensitivity against permissiveness, and modest changes in τ can alter counts once dispersion sits near the boundary (as for DiffSBDD at τ ≥ 0.15). The analysis does not replace hypothesis tests on per-metric noise models; it operationalizes a reproducible stress protocol. All models were evaluated with **n = 20** molecules per pocket and condition (or the configured panel default), which stabilizes proportion-based metrics but underpowers rare-event estimation and may not match production sample budgets. **Pocket2Mol coverage** (13/47) introduces a **confound**: brittleness on covered pockets describes only targets where reference-pocket generation succeeded, so direct numerical comparison of brittle fractions to DiffSBDD must be interpreted as a **joint** statement about coverage and dispersion, not a matched-subset claim unless analysts explicitly harmonize denominators. The featurization suite is not exhaustive—alternative orderings, noise kernels, side-chain packing assumptions, or **meaningful** pocket perturbations (for example physico-chemically motivated mutations) will surface additional failure modes [4]. **Future work** should extend the panel in size and chemical diversity, incorporate meaningful stressors that need not preserve featurization identity but remain biologically plausible, and test **mitigation** strategies such as featurization-invariant architectures [6], ensemble decoding, uncertainty quantification tied to coordinate error, and training-time augmentation aligned with experimental PDB variability. The benchmark currently evaluates two model architectures; extension to additional models including TargetDiff [9] and ResGen [10] is needed to assess whether brittleness patterns generalize across the SBDD model landscape.

---

## Methods

**Featurization stress protocol.** Each pocket was exposed to four perturbation tags treated as chemistry-preserving **featurization stressors**: (1) *atom_shuffle*—pseudo-random permutation of atom ordering within the extracted pocket; (2) *coordinate_jitter*—additive bounded noise to atomic coordinates; (3) *crop_radius_plus_1.5*—expansion of the pocket crop radius by 1.5 Å relative to baseline; (4) *crop_radius_minus_1.5*—contraction by 1.5 Å. Crop adjustments change which heavy atoms enter the pocket tensor and are therefore **not** strict mathematical invariances of a fixed atom set; we treat them as realistic pocket-*definition* perturbations alongside ordering and coordinate noise. The unperturbed pocket defined the *original* reference condition for each target.

**Brittleness flag and dominant driver.** For each pocket *p* and model *m*, let **x**(*k*, *c*) denote metric *k* on condition *c*, with *c* ranging over the four featurization-stress tags (the original condition is excluded from the dispersion). Let σ̂*k* be the sample standard deviation of {*x*(*k*, *c*)} across those four conditions. The pair (*p*, *m*) is flagged **brittle** if σ̂*k* > τ for **any** numeric summary metric *k* in the evaluation table. For visualisation (Fig. 1), the **dominant driver** at fixed τ is the metric *k* with largest σ̂*k* among those with σ̂*k* > τ (ties broken lexicographically).

**Brittleness rate** is the fraction of distinct (*p*, *m*) pairs with at least one brittle flag among **covered** pockets (reference-pocket valid count > 0). **Coverage** is the fraction of panel pockets that are covered for a given model. Thresholds τ ∈ {0.05, 0.10, 0.15, 0.20} are reported unless noted otherwise (Fig. 5; Table 1). We highlight τ = 0.10 as a conservative operational default: it lies strictly inside the plateau where DiffSBDD remains **fully** brittle on this panel (47/47) before classifications relax at τ = 0.15, aligns with the |Δ| > 0.10 responsiveness cutoff used for meaningful mutations, and represents a ten-point swing on unitless proportion summaries such as validity or QED without requiring distributional *p*-values we do not estimate here. Rates at τ = 0.05 provide a stricter sensitivity check (Fig. 5).

**Pocket panel construction.** Forty-seven protein–ligand complexes were drawn from the RCSB Protein Data Bank according to the benchmark configuration shipped with the evaluation repository [4].

**DiffSBDD setup.** DiffSBDD sampling used the repository adapter, specified checkpoints, and configuration defaults for timesteps and decoding [1]. For each pocket and each condition, the model generated *n* = **20** molecules (panel default); validity, uniqueness, valid-molecule counts, mean and standard deviation of QED, mean and standard deviation of synthetic accessibility [5], and allied tabulated fields were computed with RDKit-backed scripts shared across models [6].

**Pocket2Mol setup.** Pocket2Mol inference used the project bridge to `sample_for_pdb.py`, pretrained weights, and a reduced-beam configuration suitable for local GPU memory, with PYTHONPATH set to the Pocket2Mol repository root [2]. The same forty-seven pockets and perturbation tags were applied; *n* matched the panel comparator unless otherwise noted in extended data (https://github.com/Marooncoloredchair/PocketBench).

**Vina docking (optional).** When enabled in the driver configuration, each accepted RDKit 3D conformer is scored with **AutoDock Vina** [7] in a **20 Å** cube centered on **`pocket.ligand_centroid`** (from the same ligand anchor used for pocket extraction). Receptors are exported as PDB from the **heavy atoms present in the pocket tensor**, then converted to **PDBQT** with **Meeko** [8] as the primary path (Open Babel or MGLTools `prepare_receptor4.py` as fallbacks); ligands use **Meeko** preparation where available. The reported quantity is the **best** (most negative) binding affinity (kcal/mol) per molecule. **Vina** and **Meeko** are **optional** dependencies (`pip install vina meeko` or `pip install -e ".[vina]"`); runs without them leave docking scores empty.

**Metrics.** *Validity* is the fraction of sampled molecules that are graph-valid and chemically parsed as connected products under project rules. *Uniqueness* is the fraction of unique canonical SMILES among valid products. *Synthetic accessibility* follows RDKit’s SA score as implemented in the pipeline [5]. Brittleness aggregates use all numeric metric columns exported by the driver unless a subset is specified for sensitivity analysis.

**Statistical validation (supplement).** `analysis/statistical_validation.py` exports `paper/statistical_validation.csv`.

We retain paired pocket-level Wilcoxon signed-rank tests comparing validity under each invariant stress versus the shared reference pocket.

To ask whether coordinate jitter induces any move in pooled validity irrespective of sign, define d_p as the absolute difference between jitter conditional validity and original-pocket validity on pocket p. A one-sample Student t-test contrasts H0: expected d equals zero with H1: expected d exceeds zero; we report sample mean absolute deviation, a symmetric two-sided 95% CI for E[d], and the one-sided p-value (matching `onesample_t_mean_abs_validity_delta_jitter` in `statistical_validation.csv`).

Operational validity volatility across exactly the quartet of invariant stresses (`atom_shuffle`, `coordinate_jitter`, `crop_radius_plus_1.5`, `crop_radius_minus_1.5`) is summarized per pocket with `numpy.std(..., ddof=0)`, matching brittle-flag internals. Across pockets Wilcoxon's signed-rank procedure tests median sigma > 0 and reports the pooled median with *p* (`wilcoxon_validity_sigma_across_four_stresses`).

Brittleness rates versus fixed tau recycle `flag_invariant_brittleness` (Fig. 5); bootstrap resampling of covered-pocket labels with replacement yields the brittle-fraction point estimate plus the 2.5-97.5 percentile envelope (`bootstrap_brittleness_rate` rows).

Pure label permutations of fixed four-vector validity tuples leave sigma unchanged unless values trade pockets, so permutation evidence adopts Monte Carlo quartet resampling: pool the observable 47x4 validity tableau, redraw four IID samples with replacement for each pseudopocket row thousands of times, recompute sigma, and tally how often observed pocket sigma clears the unstructured 95th percentile benchmark (`permutation_iid_pool_validity_sigma_vs_p95_null`). Small fractions imply observed cohesion seldom reaches the unstructured upper-tail benchmark; interpret alongside Wilcoxon and tau classifications.


**Comparative analysis and figures.** Cross-model coverage, brittleness-on-covered, and validity summaries were generated with `analysis/compare_models.py`; per-pocket dominant drivers with `analysis/dominant_brittleness_metric.py` (https://github.com/Marooncoloredchair/PocketBench). Line plots, bar charts, and CSV summaries were derived from per-condition metrics and robustness summaries (Figs. 1–6 and Table 1).

**Code and data.** The benchmark pipeline, configuration files, and analysis scripts are available at https://github.com/Marooncoloredchair/PocketBench. Frozen metrics CSVs for all reported runs are included in the repository under `data/results/`. Large generation archives will be deposited on Zenodo with a citable DOI at release; until then, contact the corresponding author for bulk SDF access if the repository copy is insufficient.

---

## Figure legends (order matches `main.tex` / compiled PDF)

**Figure 1 — Dominant brittleness-driving metric (DiffSBDD, τ = 0.10).** Each row is a covered pocket; bar length is the standard deviation of the *dominant* summary column (largest dispersion across four stress conditions among columns with σ̂*k* > τ). Colors encode the column identity. Data: `paper/dominant_brittleness_metric_diffsbdd.csv`; script: `analysis/dominant_brittleness_metric.py`.

**Figure 2 — DiffSBDD versus Pocket2Mol reliability overview.** Coverage, brittleness on covered pockets versus τ, and mean original-pocket validity on shared covered pockets. Script: `analysis/compare_models.py`.

**Figure 3 — Validity for 3RFM under original and featurization stress conditions.** Four stress conditions plus original; matched sampling otherwise.

**Figure 4 — Synthetic accessibility under featurization stress.** Mean SA per pocket: original-pocket mean versus mean over four stress conditions (DiffSBDD).

**Figure 5 — Brittleness rate versus metric standard-deviation threshold (DiffSBDD).** Optional boxed note: ten-pocket subset, τ ∈ {0.05, 0.10}, brittleness 1.0 for both *n* = 20 and *n* = 50 (`paper/nsamples_sensitivity.csv`). Script: `analysis/threshold_sensitivity.py`.

**Figure 6 — Meaningful pocket mutations (DiffSBDD).** Twenty pockets; **32/40** pocket–mutation pairs responsive; interpretability stats as in main text. `paper/meaningful_perturbation_summary.csv`, `paper/figures/meaningful_deltas_heatmap.pdf`.

**Table 1 — Brittleness rate versus threshold (DiffSBDD, covered *N* = 47).** τ ∈ {0.05, 0.10, 0.15, 0.20}.

---

## References (order matches first citation in `main.tex` with `unsrtnat`)

1. Schneuing, A. *et al.* Structure-based drug design with equivariant diffusion models. *Preprint at arXiv* arXiv:2210.13695 (2022).
2. Peng, X. *et al.* Pocket2Mol: efficient molecular sampling based on 3D protein pockets. *Proc. Int. Conf. Mach. Learn.* (ICML, 2022).
3. Buttenschoen, M. *et al.* PoseCheck: generative models for 3D structure-based drug design produce unrealistic poses. *Preprint at arXiv* arXiv:2308.07413 (2023).
4. Berman, H. M. *et al.* The Protein Data Bank. *Nucleic Acids Res.* **28**, 235–242 (2000).
5. Ertl, P. & Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *J. Cheminform.* **1**, 8 (2009).
6. Landrum, G. *et al.* RDKit: open-source cheminformatics (rdkit.org, 2023).
7. Eberhardt, J., Santos-Martins, D., Tillack, A. F. & Forli, S. AutoDock Vina 1.2.0: new docking methods, expanded force field, and python bindings. *J. Chem. Inf. Model.* **61**, 3891–3898 (2021).
8. Ioannidis, J. *et al.* Meeko: preparation of small molecules for AutoDock (github.com/forlilab/Meeko, 2023).
9. Guan, J. *et al.* 3D equivariant diffusion for target-aware molecule generation and affinity prediction. *Proc. Int. Conf. Learn. Represent.* (ICLR, 2023).
10. Zhang, O. *et al.* ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling. *Nat. Mach. Intell.* **5**, 1020–1030 (2023).
