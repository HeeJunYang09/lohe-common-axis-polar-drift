"""Matplotlib style helpers for the manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler


SINGLE_COLUMN_WIDTH = 3.35
DOUBLE_COLUMN_WIDTH = 7.0


def set_manuscript_matplotlib_style() -> None:
    """
    Apply the shared Matplotlib settings used by the manuscript figures.

    The settings favor readability in the current one-column manuscript PDF,
    vector PDF output, embedded TrueType fonts, and mathtext labels rather than
    Unicode symbols.
    """
    colors = [
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    ]
    linestyles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]

    mpl.rcParams.update(
        {
            "figure.figsize": (4.8, 3.3),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "axes.linewidth": 0.9,
            "lines.markersize": 4.8,
            "lines.markeredgewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.top": True,
            "ytick.right": True,
            "axes.formatter.use_locale": False,
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (-3, 3),
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.borderpad": 0.25,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.8,
            "axes.prop_cycle": cycler(color=colors) + cycler(linestyle=linestyles),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
        }
    )


def apply_paper_style() -> None:
    """Apply the larger, interactive manuscript-figure style."""
    set_manuscript_matplotlib_style()
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "figure.constrained_layout.use": False,
        }
    )


def apply_figure1_style_for_fast_locking() -> None:
    """Backward-compatible wrapper for Figure 1."""
    set_manuscript_matplotlib_style()


def apply_figure2_style_for_phase_gap() -> None:
    """Backward-compatible wrapper for Figure 2."""
    set_manuscript_matplotlib_style()


def apply_figure3_style_for_slow_drift() -> None:
    """Backward-compatible wrapper for Figure 3."""
    set_manuscript_matplotlib_style()


def apply_figure4_style_for_drift_rate() -> None:
    """Backward-compatible wrapper for Figure 4."""
    set_manuscript_matplotlib_style()


def apply_figure5_style_for_hitting_time() -> None:
    """Backward-compatible wrapper for Figure 5."""
    set_manuscript_matplotlib_style()


def cm_to_inch(cm: float) -> float:
    """Convert centimeters to inches."""
    return cm / 2.54


def manuscript_figsize(width: str = "wide", aspect: float = 0.66) -> Tuple[float, float]:
    """Return manuscript-oriented figure size presets."""
    if width == "single":
        w = cm_to_inch(8.5)
    elif width == "onehalf":
        w = 5.0
    elif width == "wide":
        w = 6.2
    elif width == "double":
        w = DOUBLE_COLUMN_WIDTH
    else:
        raise ValueError("width must be one of: single, onehalf, wide, double")
    return (w, w * aspect)


def single_column_size(height: float = 2.4) -> Tuple[float, float]:
    """Return a single-column figure size in inches."""
    return (SINGLE_COLUMN_WIDTH, height)


def double_column_size(height: float = 3.2) -> Tuple[float, float]:
    """Return a double-column figure size in inches."""
    return (DOUBLE_COLUMN_WIDTH, height)


def save_figure(fig, basename: str, outdir: str | Path) -> tuple[Path, Path]:
    """Save a figure as vector PDF and high-resolution PNG."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{basename}.pdf"
    png_path = outdir / f"{basename}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    return pdf_path, png_path


def save_figure_all_formats(
    fig,
    outdir: str | Path,
    basename: str,
    dpi: int = 600,
) -> tuple[Path, Path, Path]:
    """Save a figure as PDF, high-resolution PNG, and EPS."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{basename}.pdf"
    png_path = outdir / f"{basename}.png"
    eps_path = outdir / f"{basename}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(eps_path, bbox_inches="tight")
    return pdf_path, png_path, eps_path


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.02) -> None:
    """Place a compact panel label such as '(a)' in axes coordinates."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def format_axes(ax, grid: bool = False) -> None:
    """Apply common axis styling after data-specific labels are set."""
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, color="0.88", linewidth=0.5)


def color_cycle() -> tuple[str, ...]:
    """Return the colorblind-friendly manuscript color cycle."""
    return (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    )
