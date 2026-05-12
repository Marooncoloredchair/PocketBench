# Results (stub)

Draft text for the Results section. Numbers are synced from `data/results/metrics_flagged__run1778460947.csv`, `metrics_per_condition__run1778460947.csv`, and `robustness_summary__run1778460947.csv` by `paper/table1.py`.

## Brittleness headline

Under the pipeline's invariant perturbations (atom shuffle, coordinate jitter ($\sigma=0.1$ Å), crop radius $\pm 1.5$ Å) and brittleness threshold (std $> 0.05$ across invariant tags), **5/5** pocket–model pairs were flagged brittle, i.e. **brittleness rate = 1.00**.

## Validity under coordinate jitter

Paired by pocket (validity under coordinate jitter minus validity on the original input), the mean change across the five pockets was **-0.120** (negative means validity **decreased** under jitter; here the average decrease is **0.120** in absolute terms).

## Synthetic accessibility (SA) drift

Across pockets, the mean change in mean SA between the original condition and the mean over invariant perturbations was **+0.238** on average (from `robustness_summary`: column `mean_sa_mean_delta`). In plain terms, mean SA **increased** relative to the original pocket under the invariant suite, aggregated as reported in the summary table.

## Per-pocket brittleness

| Pocket | Brittle | Notes (from flagged CSV) |
|--------|---------|---------------------------|
| 3RFM | Yes | n_valid:std=2.9155;validity:std=0.1458;n_unique_valid:std=2.9155;mean_sa:std=0.1861;std_sa:std=0.2047 |
| 5NDU | Yes | n_valid:std=1.2990;validity:std=0.0650;n_unique_valid:std=1.2990;mean_qed:std=0.0696;std_qed:std=0.0638;mean_sa:std=0.6782;std_sa:std=0.2154 |
| 2GM1 | Yes | n_valid:std=0.7071;n_unique_valid:std=0.7071;mean_qed:std=0.0842;mean_sa:std=0.3174;std_sa:std=0.1427 |
| 4W9W | Yes | n_valid:std=2.6810;validity:std=0.1340;n_unique_valid:std=2.6810;mean_sa:std=0.3326;std_sa:std=0.2880 |
| 2BUJ | Yes | n_valid:std=0.8292;n_unique_valid:std=0.8292;mean_qed:std=0.0728;mean_sa:std=0.3422;std_sa:std=0.2066 |
