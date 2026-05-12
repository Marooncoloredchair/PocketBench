#!/usr/bin/env python3
"""Build LaTeX Table 1 (booktabs) and Results stub from run 1778460947 metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FLAGGED = ROOT / "data" / "results" / "metrics_flagged__run1778460947.csv"
SUMMARY = ROOT / "data" / "results" / "robustness_summary__run1778460947.csv"
OUT_TEX = Path(__file__).resolve().parent / "table1_brittleness.tex"
OUT_MD = Path(__file__).resolve().parent / "results_stub.md"


def _fmt_pm(mean: float, std: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def _fmt_scalar(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def build_tex() -> str:
    flagged = pd.read_csv(FLAGGED)
    summary = pd.read_csv(SUMMARY)

    brittle = (
        flagged.groupby("pocket_id", sort=False)["brittle_invariant"]
        .first()
        .to_dict()
    )

    lines: list[str] = []
    lines.append("% Requires: \\usepackage{booktabs,amssymb}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Pocket & Validity orig. & Validity inv.\\ mean$\\pm$std & "
        "QED orig. & QED inv.\\ mean$\\pm$std & SA orig. & SA inv.\\ mean$\\pm$std & Brittle \\\\"
    )
    lines.append("\\midrule")

    for _, row in summary.iterrows():
        pid = str(row["pocket_id"])
        is_brittle = bool(brittle.get(pid, False))
        v_o = float(row["validity_original"])
        v_m = float(row["validity_inv_mean"])
        v_s = float(row["validity_inv_std"])
        q_o = float(row["mean_qed_original"])
        q_m = float(row["mean_qed_inv_mean"])
        q_s = float(row["mean_qed_inv_std"])
        s_o = float(row["mean_sa_original"])
        s_m = float(row["mean_sa_inv_mean"])
        s_s = float(row["mean_sa_inv_std"])

        cells = [
            pid,
            _fmt_scalar(v_o),
            _fmt_pm(v_m, v_s),
            _fmt_scalar(q_o),
            _fmt_pm(q_m, q_s),
            _fmt_scalar(s_o),
            _fmt_pm(s_m, s_s),
            r"\checkmark" if is_brittle else r"$\times$",
        ]
        line = " & ".join(cells) + r" \\"
        if is_brittle:
            line = " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def build_results_stub() -> str:
    flagged = pd.read_csv(FLAGGED)
    per = pd.read_csv(ROOT / "data" / "results" / "metrics_per_condition__run1778460947.csv")
    summary = pd.read_csv(SUMMARY)

    n_pairs = flagged[["pocket_id", "model_name"]].drop_duplicates().shape[0]
    brittle_pairs = (
        flagged.loc[flagged["brittle_invariant"] == True, ["pocket_id", "model_name"]]
        .drop_duplicates()
        .shape[0]
    )
    rate = brittle_pairs / n_pairs if n_pairs else 0.0

    orig = per[per["perturbation_tag"] == "original"][["pocket_id", "validity"]].set_index(
        "pocket_id"
    )["validity"]
    jitter = per[per["perturbation_tag"] == "coordinate_jitter"][
        ["pocket_id", "validity"]
    ].set_index("pocket_id")["validity"]
    delta = (jitter - orig).reindex(orig.index)
    mean_validity_drop_jitter = float(delta.mean())

    mean_sa_delta = float(summary["mean_sa_mean_delta"].mean())
    if mean_sa_delta > 0.05:
        sa_drift = "increased"
    elif mean_sa_delta < -0.05:
        sa_drift = "decreased"
    else:
        sa_drift = "mixed (near zero mean change)"

    lines: list[str] = []
    lines.append("# Results (stub)\n")
    lines.append(
        "Draft text for the Results section. Numbers are synced from "
        "`data/results/metrics_flagged__run1778460947.csv`, "
        "`metrics_per_condition__run1778460947.csv`, and "
        "`robustness_summary__run1778460947.csv` by `paper/table1.py`.\n"
    )
    lines.append("## Brittleness headline\n")
    lines.append(
        f"Under the pipeline's invariant perturbations (atom shuffle, coordinate jitter "
        f"($\\sigma=0.1$ Å), crop radius $\\pm 1.5$ Å) and brittleness threshold "
        f"(std $> 0.05$ across invariant tags), **{brittle_pairs}/{n_pairs}** pocket–model "
        f"pairs were flagged brittle, i.e. **brittleness rate = {rate:.2f}**.\n"
    )
    lines.append("## Validity under coordinate jitter\n")
    lines.append(
        f"Paired by pocket (validity under coordinate jitter minus validity on the original "
        f"input), the mean change across the five pockets was **{mean_validity_drop_jitter:+.3f}** "
        f"(negative means validity **decreased** under jitter; here the average decrease is "
        f"**{-mean_validity_drop_jitter:.3f}** in absolute terms).\n"
    )
    lines.append("## Synthetic accessibility (SA) drift\n")
    lines.append(
        f"Across pockets, the mean change in mean SA between the original condition and "
        f"the mean over invariant perturbations was **{mean_sa_delta:+.3f}** on average "
        f"(from `robustness_summary`: column `mean_sa_mean_delta`). "
        f"In plain terms, mean SA **{sa_drift}** relative to the original pocket under "
        f"the invariant suite, aggregated as reported in the summary table.\n"
    )
    lines.append("## Per-pocket brittleness\n")
    lines.append("| Pocket | Brittle | Notes (from flagged CSV) |")
    lines.append("|--------|---------|---------------------------|")
    for pid in summary["pocket_id"]:
        sub = flagged[flagged["pocket_id"] == pid].iloc[0]
        b = bool(sub["brittle_invariant"])
        note = str(sub.get("brittleness_note", "")).replace("|", "\\|")
        lines.append(f"| {pid} | {'Yes' if b else 'No'} | {note} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not FLAGGED.is_file():
        raise FileNotFoundError(FLAGGED)
    if not SUMMARY.is_file():
        raise FileNotFoundError(SUMMARY)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(build_tex(), encoding="utf-8")
    OUT_MD.write_text(build_results_stub(), encoding="utf-8")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
