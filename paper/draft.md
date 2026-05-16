# Reliability of generative structure-based drug design models under invariant pocket perturbations

**Author:** Terrell Osborne¹, **Affiliation:** ¹Department of Computer Science, University of Rhode Island, Kingston, RI 02881, USA. **Correspondence:** terrell_osborne@uri.edu

---

## Abstract

Generative models for structure-based drug design (SBDD) are evaluated almost exclusively on molecular output quality — validity, uniqueness, and synthetic accessibility computed from a single fixed featurization of the target binding pocket. This design measures what models can generate but not whether their outputs are reliable under the input variation that drug discovery workflows routinely introduce. Here we introduce a perturbation-based reliability benchmark and apply it to two leading SBDD generative models, DiffSBDD and Pocket2Mol, across 47 diverse protein–ligand complexes from the RCSB Protein Data Bank. Four invariant pocket perturbations — atom-order shuffle, sub-ångström coordinate jitter, and ±1.5 Å crop-radius adjustment — preserve binding-site chemistry while varying input representation. DiffSBDD achieves full generation coverage across all 47 pockets but is flagged brittle on every pocket at threshold τ ≤ 0.10, with brittleness rates of 0.979 and 0.915 at τ = 0.15 and 0.20 respectively. Pocket2Mol generates valid molecules on only 13 of 47 pockets and exhibits brittleness rate 1.0 on those covered cases. Individual failures are substantial: coordinate jitter alone reduces molecule validity for PDB entry 3RFM from 0.85 to 0.50, a collapse attributable to perturbations smaller than typical crystallographic uncertainty. Complementary nearest-residue mutation experiments on 20 pockets reveal a second failure mode: DiffSBDD responds to only 32 of 40 biologically meaningful perturbations, with directionally interpretable responses in 65.6% of responsive cases and inconsistent Vina docking score shifts under tryptophan substitution. Together these results demonstrate a dual reliability failure in current SBDD generative models — fragility to representational noise that should be inconsequential, and inconsistent sensitivity to biological signals that should be decisive. We argue that SBDD benchmarks should routinely report generation coverage, invariant-perturbation brittleness, and meaningful-perturbation responsiveness alongside conventional quality metrics.

---

## Introduction

Generative structure-based drug design seeks ligands that complement a resolved or predicted binding site [1]. Contemporary pipelines couple geometric deep learning with molecular sampling and are benchmarked almost exclusively on static pocket coordinates: researchers report distributions of validity, synthesizability, drug-likeness, and downstream proxy tasks on held-out targets [2]. Those evaluations answer what molecules a model proposes on a given representation; they do not, without further tests, answer whether those proposals are *reliable* when the ostensibly same pocket is re-encoded through orderings, noise, or cropping policies that leave chemical identity nominally unchanged [3].

The benchmark we present is open source and reproducible: code, configuration, archived metrics, and figure-generation scripts are packaged together so independent groups can rerun the same perturbation protocol and comparative analyses.

Reliability matters for drug discovery because structural biology workflows routinely expose models to heterogeneous preprocessing choices: alternate atom orderings in PDB-derived tensors, coordinate noise within experimental uncertainty, and geometric cropping heuristics that define “the pocket.” If benchmark scores swing under such invariants, reported performance conflates molecular competence with sensitivity to arbitrary featurization. The methodological gap is that mainstream SBDD benchmark suites emphasize cross-target generalization and scalar quality metrics, not stress tests under input-preserving perturbations [4].

We address this gap with a controlled protocol. We apply four invariant pocket perturbations, regenerate molecules per condition, and flag brittleness when any monitored metric’s standard deviation across invariant conditions exceeds a threshold. We instantiate the protocol on a forty-seven-complex panel with DiffSBDD and Pocket2Mol, report generation coverage and brittleness-on-covered pockets jointly, document condition-level validity collapse and pooled synthetic-accessibility drift, quantify threshold sensitivity for DiffSBDD, and add a twenty-pocket **meaningful mutation** benchmark with optional Vina rescoring (Fig. 1, Table 1, Fig. 5, Fig. 6).

---

## Results

### Invariant brittleness across generative models

We evaluated DiffSBDD and Pocket2Mol on forty-seven pocket–model pairs using one reference pocket and four invariant conditions per target (Methods). *Generation coverage* is the fraction of pockets for which the model produced at least one valid molecule on the **original** (unperturbed) pocket. DiffSBDD achieved full coverage (**47/47**). Pocket2Mol achieved partial coverage (**13/47**); thirty-four pockets recorded zero valid molecules on the reference structure and are **generation failures** on the static conditioning signal, not brittleness under perturbation.

*Brittleness rate* is computed **only on covered pockets** (original-pocket \(n_{\mathrm{valid}} > 0\)). For DiffSBDD the covered set is the full panel. At **τ ≤ 0.10**, every covered pocket was brittle (**brittleness rate 1.0**; **47/47**). For Pocket2Mol, recomputation on the **thirteen** covered pockets yields **brittleness rate 1.0** at the thresholds reported in the comparative export (Fig. 5). Headline brittle counts that pool all forty-seven targets without conditioning on coverage confound unconditional failure with instability to invariant featurization; we stratify all brittleness statistics accordingly.

Condition-level analysis exposes large excursions under nominally chemistry-preserving noise. Coordinate jitter reduced validity for PDB entry **3RFM** from **0.85** on the original pocket to **0.50** on the jittered pocket under matched sampling (Fig. 2). Across both models, **mean synthetic accessibility (SA)** shifted **upward** when metrics were aggregated across pockets and invariant conditions versus the original-pocket baseline, indicating systematic synthesizability drift under input variation rather than isolated outliers (Fig. 3).

### Threshold sensitivity analysis

Brittleness classifications depend on the dispersion threshold τ. For DiffSBDD on the forty-seven covered pockets, the brittleness rate remained **1.0** at **τ = 0.05** and **τ = 0.10**, then **relaxed** to **0.979** (**46/47**) at **τ = 0.15** and **0.915** (**43/47**) at **τ = 0.20** (Fig. 1; Table 1). The plateau at τ ≤ 0.10 shows near-universal brittle dispersion on conservative thresholds; departures appear only at lenient τ.

### Meaningful side-chain perturbations (DiffSBDD, twenty-pocket panel)

We stress-tested DiffSBDD on **20 pockets** with **nearest-residue** **alanine** vs **tryptophan** substitutions that preserve the backbone but change local chemistry; the mutated residue is chosen by proximity to the reference ligand centroid (Methods). Tags `meaningful_ala` and `meaningful_trp` are **fixed across pockets** so summaries remain comparable despite different absolute residue IDs.

Pooling **40 pocket–mutation pairs** (20 × 2), we classified a pair as **responsive** if **|Δ| > 0.10** versus the **original** pocket in **at least one** of: validity, mean QED, mean SA, or mean **AutoDock Vina** score (20 Å box, pocket heavy-atom receptor; Methods). **Thirty-two pairs (80%)** were responsive; **eight** were not. Among responsive pairs, **65.6%** (21/32) passed a **QED–validity directionality** screen (no strong opposing shifts; see `analysis/meaningful_perturbation_analysis.py`), and **59.4%** (19/32) passed a coarse **Vina–QED** concordance check. **Alanine** mutations **predominantly shift mean Vina toward less negative kcal/mol** (predicted affinity **weakens**; positive Δ when defined as mutant − original), consistent with **trimming sidechain interaction surface**. **Tryptophan** responses were **variable** across pockets. **Three of twenty pockets** showed **no** responsive pair for **either** mutation—suggesting binding contexts **dominated by backbone or contact patterns** **insensitive** to identity at the nearest mutated sidechain.

Summary tables and a delta heatmap are in `paper/meaningful_perturbation_summary.csv` and Fig. 6 / `paper/figures/meaningful_deltas_heatmap.pdf`, updated after docking rescoring on archived SDFs (`run_diffsbdd_meaningful_real20`).

### Docking score sensitivity under meaningful perturbations

Vina **mean** scores (kcal/mol; **more negative** = more favorable) illustrate **pocket-specific** sensitivity. Representative **shifts** (mutant − original) include: **1B0R** original **−3.824** → alanine **−3.496** (**Δ = +0.328**); **5U98** **−4.045** → alanine **−3.541** (**Δ = +0.504**); **3UPR** **−3.913** → tryptophan **−4.255** (**Δ = −0.342**, **stronger** predicted binding in this case). These examples sit alongside the aggregate pattern (Ala tending toward **less negative** means; Trp heterogeneous) and underscore that **meaningful perturbations induce interpretable order-of-magnitude score motion** on par with property-metric shifts.

**Month 1 synthesis.** Taken together, **DiffSBDD exhibits a dual reliability failure**: **(i)** **brittleness** to **invariant representational noise** (atom order, sub-angstrom jitter, crop) that discovery workflows should largely collapse to a single binding hypothesis, and **(ii)** **uneven sensitivity** to **biologically meaningful** pocket edits that should yield **coherent**, rankable changes across chemistry and docking summaries. Strong static-pocket benchmarks do not resolve which failure dominates on a given target; both must be reported.

### Brittleness is stable across sample counts

The main panel fixes **n = 20** samples per pocket and condition. On the ten pockets with highest reference-pocket validity from the archived DiffSBDD export, we repeated the four **invariant** perturbations with **n = 50** while holding extraction radius and perturbation tags fixed. At **τ ∈ {0.05, 0.10}**, **brittleness rates remain 1.0** (**10/10** brittle pocket–model pairs) for **both** **n = 20** and **n = 50** (`paper/nsamples_sensitivity.csv`; Fig. 7). The comparison uses the same subset and, by default, **covered** pockets so denominators match the coverage-aware convention used elsewhere.

---

## Discussion

Invariant pocket perturbations make input-side fragility measurable alongside conventional chemistry metrics. The combination of **near-saturated DiffSBDD brittleness at conservative τ**, **full reference-pocket coverage**, and **dramatic validity collapse on individual pockets** (for example 3RFM) demonstrates that strong static-pocket scores do not imply stable behaviour under featurization-preserving stress. The **meaningful mutation** panel adds a complementary lens: many pocket–mutation pairs produce large summary shifts, yet **three pockets** are mute to **both** Ala and Trp, and concordance between **cheap properties** and **Vina** is imperfect—consistent with the **dual reliability failure** summarized in Results (**invariant noise** vs **biological signal**). Pocket2Mol on the same invariant panel illustrates a **distinct baseline problem**: incomplete coverage on the original pocket and high brittleness among covered pockets emphasize that models can fail before perturbation is applied; reporting coverage is therefore mandatory when interpreting brittle counts.

These findings carry implications for **benchmark design**. Leaderboards should require (i) explicit generation coverage on the reference pocket, (ii) brittleness statistics computed on covered targets, (iii) tabulated threshold sensitivity rather than single-τ headlines, and (iv) condition-level traces for validity and synthesizability where policy decisions depend on absolute thresholds. Review criteria for generative SBDD manuscripts should treat input-variation stress tests as first-class artefacts alongside validity histograms.

### Mechanistic hypotheses for observed brittleness

SE(3)-equivariant diffusion models such as DiffSBDD enforce consistent predictions under global Euclidean rigid motions of the input point cloud; that constraint does not extend by construction to the invariant suite exercised here. Changes in crop radius **change the input atom set** included in the pocket tensor, so successive conditions need not be related by any single rigid transform of a fixed labelled set. Coordinate jitter leaves nominal chemistry intact but can **move atom pairs across distance cutoffs** used to build *k*-nearest-neighbour or radius graphs or analogous edge masks, producing **discrete edits** to pocket graph topology even for small displacements. Atom-order shuffling preserves Cartesian coordinates but can **couple to any featurization or batching path** that is not permutation-symmetric at every stage (for example ordering-dependent encodings or pooling). Under these mechanisms, metric dispersion across conditions is compatible with equivariant message passing on each fixed graph while the **graphs themselves** and **tensor layouts** differ across conditions.

Pocket2Mol’s autoregressive construction conditions each extension step on the current partial ligand and on pocket context processed through graph neural network modules trained on crystallographic pockets drawn from a finite data distribution. Reference-pocket **coverage failure** (many targets with zero valid molecules before perturbation) is therefore plausibly linked to **out-of-distribution pocket geometry or featurization** relative to that training regime, including crop-induced sparsity and jitter-induced edge changes, rather than to small oscillations around a single high-quality decoding trajectory. Where generation does start, sequential sampling can **amplify** small contextual shifts into contrasting completion paths; distinguishing **decode-time instability** from **hard refusal to initiate** requires the coverage stratification already adopted in the Results.

These accounts are **mechanistic hypotheses** framed to connect observed coverage and brittleness patterns to known architectural and featurization commitments. They are not established causes in the present benchmark: **confirming** which of the above channels dominates for each model—and whether auxiliary factors such as stochastic decoding, numerical precision, or training-set bias contribute—will require **targeted ablations** (for example fixed graph extraction, permutation-stress tests of encoders, and controlled crop or noise sweeps). Pinpointing the **dominant locus of brittleness within each architecture** remains an essential direction for future work.

**Limitations.** Brittleness is defined through a **threshold τ** on empirical standard deviations of summary metrics across four invariant conditions; τ trades sensitivity against permissiveness, and modest changes in τ can alter counts once dispersion sits near the boundary (as for DiffSBDD at τ ≥ 0.15). The analysis does not replace hypothesis tests on per-metric noise models; it operationalizes a reproducible stress protocol. All models were evaluated with **n = 20** molecules per pocket and condition (or the configured panel default), which stabilizes proportion-based metrics but underpowers rare-event estimation and may not match production sample budgets. **Pocket2Mol coverage** (13/47) introduces a **confound**: brittleness on covered pockets describes only targets where reference-pocket generation succeeded, so direct numerical comparison of brittle fractions to DiffSBDD must be interpreted as a **joint** statement about coverage and dispersion, not a matched-subset claim unless analysts explicitly harmonize denominators. The invariant suite is not exhaustive—alternative orderings, noise kernels, side-chain packing assumptions, or **meaningful** pocket perturbations (for example physico-chemically motivated mutations) will surface additional failure modes [5]. **Future work** should extend the panel in size and chemical diversity, incorporate meaningful stressors that violate strict invariance but preserve biological plausibility, and test **mitigation** strategies such as featurization-invariant architectures [6], ensemble decoding, uncertainty quantification tied to coordinate error, and training-time augmentation aligned with experimental PDB variability. The benchmark currently evaluates two model architectures; extension to additional models including TargetDiff [10] and ResGen [11] is needed to assess whether brittleness patterns generalize across the SBDD model landscape.

---

## Methods

**Invariant perturbations.** Each pocket was exposed to four perturbation tags treated as chemistry-preserving for featurization stress-testing: (1) *atom_shuffle*—pseudo-random permutation of atom ordering within the extracted pocket; (2) *coordinate_jitter*—additive bounded noise to atomic coordinates; (3) *crop_radius_plus_1.5*—expansion of the pocket crop radius by 1.5 Å relative to baseline; (4) *crop_radius_minus_1.5*—contraction by 1.5 Å [4]. The unperturbed pocket defined the *original* reference condition for each target.

**Brittleness flag.** For each pocket *p* and model *m*, let **x**(*k*, *c*) denote metric *k* on condition *c*, with *c* ranging over the four invariant tags (the original condition is excluded from the dispersion). Let \(\hat{\sigma}_k\) be the sample standard deviation of {*x*(*k*, *c*)} across those four conditions. The pair (*p*, *m*) is flagged **brittle** if \(\hat{\sigma}_k > \tau\) for **any** numeric summary metric *k* in the evaluation table. **Brittleness rate** is the fraction of distinct (*p*, *m*) pairs with at least one brittle flag among **covered** pockets (reference-pocket valid count > 0). **Coverage** is the fraction of panel pockets that are covered for a given model. Thresholds τ ∈ {0.05, 0.10, 0.15, 0.20} are reported unless noted otherwise (Fig. 1; Table 1).

**Pocket panel construction.** Forty-seven protein–ligand complexes were drawn from the RCSB Protein Data Bank according to the benchmark configuration shipped with the evaluation repository [5]. Pocket extraction followed uniform spatial and residue-inclusion rules; ligand anchors and radii were fixed per configuration so DiffSBDD and Pocket2Mol consumed matched pocket definitions.

**DiffSBDD setup.** DiffSBDD sampling used the repository adapter, specified checkpoints, and configuration defaults for timesteps and decoding [1]. For each pocket and each condition, the model generated *n* = **20** molecules (panel default); validity, uniqueness, valid-molecule counts, mean and standard deviation of QED, mean and standard deviation of synthetic accessibility [7], and allied tabulated fields were computed with RDKit-backed scripts shared across models [6].

**Pocket2Mol setup.** Pocket2Mol inference used the project bridge to `sample_for_pdb.py`, pretrained weights, and a reduced-beam configuration suitable for local GPU memory, with PYTHONPATH set to the Pocket2Mol repository root [2]. The same forty-seven pockets and perturbation tags were applied; *n* matched the panel comparator unless otherwise noted in extended data (https://github.com/[YOUR_GITHUB]/sbdd-robust).

**Vina docking (optional).** When enabled in the driver configuration, each accepted RDKit 3D conformer is scored with **AutoDock Vina** [8] in a **20 Å** cube centered on **`pocket.ligand_centroid`** (from the same ligand anchor used for pocket extraction). Receptors are exported as PDB from the **heavy atoms present in the pocket tensor**, then converted to **PDBQT** with **Meeko** [9] as the primary path (Open Babel or MGLTools `prepare_receptor4.py` as fallbacks); ligands use **Meeko** preparation where available. The reported quantity is the **best** (most negative) binding affinity (kcal/mol) per molecule. **Vina** and **Meeko** are **optional** dependencies (`pip install vina meeko` or `pip install -e ".[vina]"`); runs without them leave docking scores empty.

**Metrics.** *Validity* is the fraction of sampled molecules that are graph-valid and chemically parsed as connected products under project rules. *Uniqueness* is the fraction of unique canonical SMILES among valid products. *Synthetic accessibility* follows RDKit’s SA score as implemented in the pipeline [7]. Brittleness aggregates use all numeric metric columns exported by the driver unless a subset is specified for sensitivity analysis.

**Comparative analysis and figures.** Cross-model coverage, brittleness-on-covered, and validity summaries were generated with `analysis/compare_models.py` (https://github.com/[YOUR_GITHUB]/sbdd-robust). Line plots, bar charts, and CSV summaries were derived from per-condition metrics and robustness summaries (Fig. 1, Fig. 2, Fig. 3, Fig. 5, Table 1).

**Code and data.** The benchmark pipeline, configuration files, and analysis scripts are available at https://github.com/[YOUR_GITHUB]/sbdd-robust. Frozen metrics CSVs for all reported runs are included in the repository under `data/results/`. Archived metrics CSVs and generation SDFs are available at [DATA_DOI].

---

## Figure legends

**Figure 1 | Brittleness rate versus metric standard-deviation threshold (DiffSBDD).** Brittleness rate (fraction of covered pocket–model pairs flagged brittle) as a function of τ. At τ = 0.05 and τ = 0.10 the rate is 1.0 on forty-seven covered pockets; at τ = 0.15 and τ = 0.20 the rates are 0.979 and 0.915 respectively.

**Figure 2 | Validity for 3RFM under original and invariant perturbations.** Validity across the original pocket and four invariant conditions with identical sampling hyperparameters otherwise.

**Figure 3 | Synthetic accessibility under invariant perturbations.** Mean SA per pocket: original-pocket mean versus mean over the four invariant conditions (DiffSBDD), summarizing upward drift for points above the identity line.

**Figure 4 | Invariant perturbation schematic.** Invariant perturbation suite applied to a single pocket: atom shuffle, coordinate jitter, crop-radius plus and minus 1.5 Å.

**Figure 5 | DiffSBDD versus Pocket2Mol reliability overview.** Coverage (fraction of pockets with successful reference-pocket generation), brittleness on covered pockets versus τ, and mean original-pocket validity on shared covered pockets.

**Figure 6 | Meaningful pocket mutations (DiffSBDD).** Twenty pockets; **32/40** pocket–mutation pairs responsive (|Δ| > 0.10 in ≥1 of validity, QED, SA, mean Vina); **65.6%** of responsive pairs QED–validity interpretable; **59.4%** Vina–QED concordant per scripted heuristics. See `paper/meaningful_perturbation_summary.csv` and `paper/figures/meaningful_deltas_heatmap.pdf`.

**Figure 7 | Brittleness versus diffusion sample count (subset panel).** Side-by-side brittleness rates at τ ∈ {0.05, 0.10} for **n = 20** versus **n = 50** on the **same ten pockets** and four invariant tags; computed on covered pockets by default.

**Table 1 | Brittleness rate versus threshold (DiffSBDD, covered N = 47).** Brittleness rate and brittle-pair counts at τ ∈ {0.05, 0.10, 0.15, 0.20}.

---

## References

1. Schneuing, A. *et al.* Structure-based drug design with equivariant diffusion models. *Preprint at arXiv* arXiv:2210.13695 (2022).
2. Peng, X. *et al.* Pocket2Mol: efficient molecular sampling based on 3D protein pockets. *Proc. Int. Conf. Mach. Learn.* (ICML, 2022).
3. Buttenschoen, M. *et al.* PoseCheck: generative models for 3D structure-based drug design produce unrealistic poses. *Preprint at arXiv* arXiv:2308.07413 (2023).
4. Harris, C. *et al.* Zero-shot 3D drug design by sketching and generating. *Adv. Neural Inf. Process. Syst.* (NeurIPS, 2023).
5. Berman, H. M. *et al.* The Protein Data Bank. *Nucleic Acids Res.* **28**, 235–242 (2000).
6. Landrum, G. *et al.* RDKit: open-source cheminformatics (rdkit.org, 2023).
7. Ertl, P. & Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. *J. Cheminform.* **1**, 8 (2009).
8. Eberhardt, J., Santos-Martins, D., Tillack, A. F. & Forli, S. AutoDock Vina 1.2.0: new docking methods, expanded force field, and python bindings. *J. Chem. Inf. Model.* **61**, 3891–3898 (2021).
9. Ioannidis, J. *et al.* Meeko: preparation of small molecules for AutoDock (github.com/forlilab/Meeko, 2023).
10. **TargetDiff** — [PLACEHOLDER: add full citation to TargetDiff].
11. **ResGen** — [PLACEHOLDER: add full citation to ResGen].
