"""Render multi-target hitting-time figures from saved diagnostics only."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lohe-additional-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from figure_style import apply_paper_style, format_axes

OUTPUT = Path(__file__).resolve().parents[1]
STEM = "fig_multi_target_hitting_time"
ETAS = (0.84, 0.82, 0.70, 0.50, 0.30)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Experiment directory containing data/ and figures/.")
    args = parser.parse_args()
    if args.output_dir:
        OUTPUT = args.output_dir.resolve()
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    generated = {OUTPUT / "figures" / f"{STEM}.{ext}" for ext in ("pdf", "png")}
    baseline = {p: digest(p) for p in OUTPUT.rglob("*") if p.is_file() and p not in generated}
    with (OUTPUT / "data" / "hitting_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (OUTPUT / "data" / "eta_summary.csv").open(newline="") as stream:
        summaries = {float(row["eta"]): row for row in csv.DictReader(stream)}

    selected = {}
    errors = {}
    for eta in ETAS:
        subset = [row for row in rows if float(row["eta"]) == eta]
        if len(subset) != 25 or any(row["threshold_reached"] != "True" for row in subset):
            raise ValueError(f"Expected 25 reached targets at eta={eta}.")
        measured = np.array([float(row["T_eta_sim"]) for row in subset])
        predicted = np.array([float(row["T_eta_pred"]) for row in subset])
        relative = np.abs(measured - predicted) / measured
        np.testing.assert_allclose(relative, [float(row["relative_error_total"]) for row in subset],
                                   rtol=1e-13, atol=1e-15)
        mean, maximum = relative.mean(), relative.max()
        np.testing.assert_allclose([mean, maximum],
                                   [float(summaries[eta]["relative_error_total_mean"]),
                                    float(summaries[eta]["relative_error_total_max"])],
                                   rtol=1e-13, atol=1e-15)
        selected[eta] = subset
        errors[eta] = (100 * mean, 100 * maximum)

    apply_paper_style()
    # Match the manuscript Figure 3/4 typography and canvas dimensions.
    plt.rcParams.update({"font.size": 17, "axes.labelsize": 19,
                         "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.5))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.94, bottom=0.28, wspace=0.28)
    colors = ("#CC79A7", "#0072B2", "#D55E00", "#009E73", "#000000")
    markers = ("o", "s", "^", "D", "v")
    for eta, color, marker in zip(ETAS, colors, markers):
        subset = selected[eta]
        axes[0].scatter([float(row["log_factor"]) for row in subset],
                        [float(row["scaled_slow_time"]) for row in subset],
                        s=46, marker=marker, edgecolors=color, facecolors="none",
                        linewidths=1.3, zorder=6, label=rf"$\eta={eta:.2f}$")
    upper = max(float(row["scaled_slow_time"]) for row in rows) * 1.045
    axes[0].plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=2.2,
                 zorder=4, label="Prediction")
    axes[0].set(xlim=(0, upper), ylim=(0, upper),
                xlabel=r"$\log[\tan\bar\phi(t_f^{\mathrm{num}})/\tan\eta]$",
                ylabel=r"$\Lambda_K(T_\eta^{\mathrm{sim}}-t_f^{\mathrm{num}})$")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1],
                   loc="upper left", handlelength=1.1,
                   handletextpad=0.4, labelspacing=0.15)
    axes[0].set_xticks([0.0, 0.4, 0.8, 1.2])
    axes[0].set_yticks([0.0, 0.4, 0.8, 1.2])

    # Categorical spacing keeps the nearby targets 0.82 and 0.84 legible.
    ascending = sorted(ETAS)
    positions = np.arange(len(ascending))
    axes[1].plot(positions, [errors[eta][0] for eta in ascending],
                 color="#0072B2", marker="o", markersize=7, linewidth=2.0,
                 linestyle="none", label="Mean", zorder=5)
    axes[1].plot(positions, [errors[eta][1] for eta in ascending],
                 color="#D55E00", marker="s", markersize=7, linewidth=2.0,
                 linestyle="none", label="Maximum", zorder=5)
    axes[1].set(xlim=(-0.2, 4.2), ylim=(0, 5),
                xlabel=r"Target polar angle $\eta$", ylabel="Relative error")
    axes[1].set_xticks(positions, [f"{eta:.2f}" for eta in ascending])
    axes[1].set_yticks(np.arange(6))
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes[1].legend(loc="upper left")
    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.set_box_aspect(1)
        format_axes(ax)
        ax.xaxis.labelpad = 4
        ax.yaxis.labelpad = 4
        ax.text(0.5, -0.25, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=17)
    axes[1].xaxis.set_minor_locator(NullLocator())

    for extension in ("pdf", "png"):
        path = OUTPUT / "figures" / f"{STEM}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=600)
        print(f"Saved {path}")
    plt.close(fig)
    changed = [str(path) for path, checksum in baseline.items() if digest(path) != checksum]
    if changed:
        raise RuntimeError(f"Existing artifacts changed: {changed}")
    print(f"PASS preserved {len(baseline)} existing artifacts; verified 125 relative errors.")
    import run_multi_target as experiment
    experiment.OUTPUT = OUTPUT
    experiment.plot(experiment.validate())
    for eta in ETAS:
        mean, maximum = errors[eta]
        print(f"eta={eta:.2f}: mean={mean:.6f}%, max={maximum:.6f}%")


if __name__ == "__main__":
    main()
