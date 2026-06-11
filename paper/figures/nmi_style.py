"""Shared Matplotlib styling for Nature Machine Intelligence-style figures (colorblind-safe palette)."""

from __future__ import annotations

from pathlib import Path

# ColorBrewer-inspired / Wong palette style for 3+ series
COLOR_DIFFSBDD = "#0072B2"
COLOR_POCKET2MOL = "#E69F00"
COLOR_TARGETDIFF = "#009E73"
COLOR_ACCENT = "#CC79A7"


def setup_rc(single_panel: bool) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def figsize_single() -> tuple[float, float]:
    return (3.5, 2.8)


def figsize_tripanel() -> tuple[float, float]:
    return (7.0, 2.8)


def save_figure(fig, out_base: Path) -> tuple[Path, Path]:
    """Write PDF and PNG (300 dpi) next to each other."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    stem = out_base.with_suffix("")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(png, format="png", bbox_inches="tight", facecolor="white", dpi=300)
    return pdf, png
