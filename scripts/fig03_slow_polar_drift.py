#%%
# Figure 03: slow polar drift and drift-rate collapse

#%%
# Imports and project paths
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common_utils import (
    compute_lambda_K,
    deterministic_frequencies,
    first_threshold_index,
    fit_logtan_slope,
    frequency_statistics,
    gaussian_frequencies,
    integrate_sphere_model,
    locking_diagnostics,
    logtan,
    make_theorem_regime_initial_condition,
    save_metadata,
    solve_locked_profile,
)
from figure_style import apply_paper_style, format_axes, save_figure_all_formats
from run_receipts import record_figure_regression

apply_paper_style()
plt.rcParams.update(
    {
        "font.size": 17,
        "axes.labelsize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    }
)


#%%
# Parameters. Change only FREQUENCY_MODE to switch figures.
RECOMPUTE = False
FREQUENCY_MODE = "deterministic"  # "deterministic" or "gaussian"
# FREQUENCY_MODE = "gaussian"  # "deterministic" or "gaussian"

N = 32
PANEL_A_PAIRS = [(8.0, 0.18), (10.0, 0.20), (12.0, 0.22)]
PANEL_B_K_VALUES = [8.0, 10.0, 12.0]
PANEL_B_SIGMA_VALUES = [0.16, 0.20, 0.24]
SEEDS = np.arange(5)
OMEGA_BAR = 0.5
THETA0 = 0.3
PHI0 = 0.85
C_INIT = 0.30
C_TOL = 5.0
T1 = 220.0
NUM_SAVE = 1800
RTOL = 1.0e-7
ATOL = 1.0e-9
FIT_PHI_MIN = 0.22
FIT_PHI_MAX = 0.78
LEGEND_FONTSIZE = 13

FIGURE_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed"
for directory in (FIGURE_DIR, CACHE_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GRID_TAG = f"{len(PANEL_B_K_VALUES)}x{len(PANEL_B_SIGMA_VALUES)}"
SEED_TAG = f"seeds{int(SEEDS[0])}-{int(SEEDS[-1])}"
if FREQUENCY_MODE == "deterministic":
    CACHE_FILE = CACHE_DIR / f"fig03_slow_polar_drift_deterministic_{GRID_TAG}.npz"
    SUMMARY_FILE = SUMMARY_DIR / f"summary_exp03_slow_polar_drift_deterministic_{GRID_TAG}.csv"
    METADATA_FILE = SUMMARY_DIR / f"metadata_exp03_deterministic_{GRID_TAG}.json"
    FIGURE_STEM = "fig3_slow_polar_drift_deterministic"
elif FREQUENCY_MODE == "gaussian":
    CACHE_FILE = CACHE_DIR / f"fig03_slow_polar_drift_gaussian_{GRID_TAG}_{SEED_TAG}.npz"
    SUMMARY_FILE = SUMMARY_DIR / f"summary_exp03_slow_polar_drift_gaussian_{GRID_TAG}_{SEED_TAG}.csv"
    SEED_METRICS_FILE = (
        SUMMARY_DIR / f"summary_exp03_slow_polar_drift_gaussian_seed_metrics_{GRID_TAG}_{SEED_TAG}.csv"
    )
    METADATA_FILE = SUMMARY_DIR / f"metadata_exp03_gaussian_{GRID_TAG}_{SEED_TAG}.json"
    FIGURE_STEM = "fig3_slow_polar_drift_gaussian"
else:
    raise ValueError(f"Unknown FREQUENCY_MODE: {FREQUENCY_MODE}")

#%%
# One full numerical experiment
def make_frequencies(sigma, seed=None):
    if FREQUENCY_MODE == "deterministic":
        return deterministic_frequencies(N=N, omega_bar=OMEGA_BAR, sigma_omega=sigma)
    if FREQUENCY_MODE == "gaussian":
        if seed is None:
            raise ValueError("Gaussian frequency mode requires a seed.")
        return gaussian_frequencies(
            N=N, omega_bar=OMEGA_BAR, sigma_omega=sigma, seed=int(seed)
        )
    raise ValueError(f"Unknown FREQUENCY_MODE: {FREQUENCY_MODE}")


def simulate(K, sigma, seed=None):
    omega = make_frequencies(sigma=sigma, seed=seed)
    stats = frequency_statistics(omega, K)
    vartheta, locked_residual = solve_locked_profile(omega, K)
    Lambda_K = compute_lambda_K(vartheta, K)
    x0, _ = make_theorem_regime_initial_condition(
        omega, K, vartheta, theta0=THETA0, phi0=PHI0, perturb_scale=C_INIT
    )
    result = integrate_sphere_model(
        omega, K, x0, t0=0.0, t1=T1, num_save=NUM_SAVE, rtol=RTOL, atol=ATOL
    )
    diag = locking_diagnostics(result.ts, result.xs, stats["omega_bar"], vartheta)
    idx = first_threshold_index(diag["E_lock"], C_TOL * stats["rho"] ** 2)
    fast_threshold_reached = idx is not None
    if idx is None:
        idx = int(np.argmin(diag["E_lock"]))
    tf_num = float(result.ts[idx])
    slope, _, fit_mask = fit_logtan_slope(
        result.ts, diag["phi_bar"], tf_num, FIT_PHI_MIN, FIT_PHI_MAX, min_points=8
    )
    Y = logtan(diag["phi_bar"])
    D_meas = -float(slope)
    D_over_Lambda = D_meas / Lambda_K
    return {
        "ts": result.ts,
        "Y": Y,
        "Y_tf": float(np.interp(tf_num, result.ts, Y)),
        "fit_mask": fit_mask,
        "seed": -1 if seed is None else int(seed),
        "K": float(K),
        "sigma": float(sigma),
        "rho": float(stats["rho"]),
        "Lambda_K": float(Lambda_K),
        "tf_num": tf_num,
        "slope_rescaled": float(slope / Lambda_K),
        "D_meas": D_meas,
        "D_meas_over_Lambda_K": float(D_over_Lambda),
        "relative_drift_error": float(abs(D_over_Lambda - 1.0)),
        "fast_threshold_reached": fast_threshold_reached,
        "locked_residual": float(locked_residual),
        "sphere_norm_error": float(result.stats["sphere_norm_error"]),
    }

#%%
# Load or generate mode-specific data
if CACHE_FILE.exists() and not RECOMPUTE:
    data = dict(np.load(CACHE_FILE))
    print("Loaded:", CACHE_FILE)
else:
    if FREQUENCY_MODE == "deterministic":
        panel_a = [simulate(*pair) for pair in PANEL_A_PAIRS]
        panel_b = [
            simulate(K, sigma)
            for K in PANEL_B_K_VALUES
            for sigma in PANEL_B_SIGMA_VALUES
        ]
    else:
        panel_a = [
            simulate(K, sigma, seed=int(SEEDS[0])) for K, sigma in PANEL_A_PAIRS
        ]
        panel_b = [
            simulate(K, sigma, seed=int(seed))
            for seed in SEEDS
            for K in PANEL_B_K_VALUES
            for sigma in PANEL_B_SIGMA_VALUES
        ]

    data = {
        "frequency_mode": np.array(FREQUENCY_MODE),
        "ts": panel_a[0]["ts"],
        "K_values": np.array([r["K"] for r in panel_a]),
        "sigma_values": np.array([r["sigma"] for r in panel_a]),
        "Y": np.stack([r["Y"] for r in panel_a]),
        "Y_tf": np.array([r["Y_tf"] for r in panel_a]),
        "fit_mask": np.stack([r["fit_mask"] for r in panel_a]),
        "Lambda_K": np.array([r["Lambda_K"] for r in panel_a]),
        "tf_num": np.array([r["tf_num"] for r in panel_a]),
        "slope_rescaled": np.array([r["slope_rescaled"] for r in panel_a]),
        "panel_b_seed": np.array([r["seed"] for r in panel_b], dtype=int),
        "panel_b_K": np.array([r["K"] for r in panel_b]),
        "panel_b_sigma": np.array([r["sigma"] for r in panel_b]),
        "panel_b_rho": np.array([r["rho"] for r in panel_b]),
        "panel_b_Lambda_K": np.array([r["Lambda_K"] for r in panel_b]),
        "panel_b_D_meas": np.array([r["D_meas"] for r in panel_b]),
        "panel_b_D_meas_over_Lambda_K": np.array(
            [r["D_meas_over_Lambda_K"] for r in panel_b]
        ),
        "panel_b_relative_drift_error": np.array(
            [r["relative_drift_error"] for r in panel_b]
        ),
        "panel_b_tf_num": np.array([r["tf_num"] for r in panel_b]),
        "panel_b_locked_residual": np.array([r["locked_residual"] for r in panel_b]),
        "panel_b_sphere_norm_error": np.array([r["sphere_norm_error"] for r in panel_b]),
        "panel_b_fast_threshold_reached": np.array(
            [r["fast_threshold_reached"] for r in panel_b], dtype=bool
        ),
    }
    np.savez_compressed(CACHE_FILE, **data)
    print("Saved:", CACHE_FILE)

#%%
# Diagnostics, summary CSV, and metadata
rows = []
for idx in range(len(data["panel_b_K"])):
    rows.append(
        {
            "frequency_mode": FREQUENCY_MODE,
            "seed": int(data["panel_b_seed"][idx]),
            "K": data["panel_b_K"][idx],
            "sigma_omega": data["panel_b_sigma"][idx],
            "rho": data["panel_b_rho"][idx],
            "Lambda_K": data["panel_b_Lambda_K"][idx],
            "D_meas": data["panel_b_D_meas"][idx],
            "D_meas_over_Lambda_K": data["panel_b_D_meas_over_Lambda_K"][idx],
            "relative_drift_error": data["panel_b_relative_drift_error"][idx],
            "t_f_num": data["panel_b_tf_num"][idx],
            "fast_threshold_reached": bool(data["panel_b_fast_threshold_reached"][idx]),
            "locked_profile_residual": data["panel_b_locked_residual"][idx],
            "sphere_norm_error": data["panel_b_sphere_norm_error"][idx],
        }
    )
with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

ratios = data["panel_b_D_meas_over_Lambda_K"]
errors = data["panel_b_relative_drift_error"]
metadata = {
    "figure": "Figure 3",
    "frequency_mode": FREQUENCY_MODE,
    "cache_file": CACHE_FILE.relative_to(PROJECT_ROOT),
    "summary_file": SUMMARY_FILE.relative_to(PROJECT_ROOT),
    "N": N,
    "panel_a_pairs": PANEL_A_PAIRS,
    "panel_b_K_values": PANEL_B_K_VALUES,
    "panel_b_sigma_values": PANEL_B_SIGMA_VALUES,
    "seeds": SEEDS if FREQUENCY_MODE == "gaussian" else [],
    "theta0": THETA0,
    "phi0": PHI0,
    "c_init": C_INIT,
    "C_tol": C_TOL,
    "rho_min": float(np.min(data["panel_b_rho"])),
    "rho_max": float(np.max(data["panel_b_rho"])),
    "D_over_Lambda_min": float(np.min(ratios)),
    "D_over_Lambda_max": float(np.max(ratios)),
    "D_over_Lambda_mean": float(np.mean(ratios)),
    "relative_drift_error_max": float(np.max(errors)),
    "relative_drift_error_mean": float(np.mean(errors)),
    "all_fast_thresholds_reached": bool(np.all(data["panel_b_fast_threshold_reached"])),
}

if FREQUENCY_MODE == "gaussian":
    seed_rows = []
    for seed in SEEDS:
        mask = data["panel_b_seed"] == seed
        seed_rows.append(
            {
                "seed": int(seed),
                "D_over_Lambda_min_seed": float(np.min(ratios[mask])),
                "D_over_Lambda_max_seed": float(np.max(ratios[mask])),
                "D_over_Lambda_mean_seed": float(np.mean(ratios[mask])),
                "relative_drift_error_max_seed": float(np.max(errors[mask])),
                "relative_drift_error_mean_seed": float(np.mean(errors[mask])),
                "rho_min_seed": float(np.min(data["panel_b_rho"][mask])),
                "rho_max_seed": float(np.max(data["panel_b_rho"][mask])),
                "all_fast_thresholds_reached_seed": bool(
                    np.all(data["panel_b_fast_threshold_reached"][mask])
                ),
            }
        )
    with SEED_METRICS_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    metadata["seed_metrics_file"] = SEED_METRICS_FILE.relative_to(PROJECT_ROOT)
    metadata["seed_metrics"] = seed_rows

save_metadata(METADATA_FILE, metadata)
print("Saved:", SUMMARY_FILE)
print("Saved:", METADATA_FILE)
if FREQUENCY_MODE == "gaussian":
    print("Saved:", SEED_METRICS_FILE)

#%%
# Plot both modes with the deterministic Figure 3 style
FIGSIZE = (7.8, 4.5)
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.095, 0.985, 0.28, 0.94, 0.34
PANEL_LABEL_Y = -0.25
PANEL_LABEL_FONTSIZE = 17
XLABEL_PAD = 4
YLABEL_PAD = 4


def plot_figure3(plot_data):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax_a, ax_b = axes
    colors = ["#0072B2", "#D55E00", "#009E73"]
    linestyles = ["-", "--", "-."]

    x_max, y_min = 0.0, 0.0
    for j, K in enumerate(plot_data["K_values"]):
        mask = plot_data["fit_mask"][j]
        X = plot_data["Lambda_K"][j] * (plot_data["ts"][mask] - plot_data["tf_num"][j])
        Y_shifted = plot_data["Y"][j, mask] - plot_data["Y_tf"][j]
        x_max = max(x_max, float(np.max(X)))
        y_min = min(y_min, float(np.min(Y_shifted)))
        ax_a.plot(
            X,
            Y_shifted,
            color=colors[j],
            linestyle=linestyles[j],
            linewidth=2.0,
            label=rf"$K={K:g},\ \sigma_\omega={plot_data['sigma_values'][j]:.2f}$",
        )
    # x_ref = np.linspace(0.0, x_max, 200)
    x_ref = np.linspace(0.0, 1.0, 200)
    ax_a.plot(
        x_ref, -x_ref, linestyle="--", color="0.25", linewidth=1.6,
        label=r"slope $-1$", zorder=1,
    )
    ax_a.set(
        xlabel=r"$X=\Lambda_K(t-t_f^{\mathrm{num}})$",
        ylabel=r"$Y(t)$",
        xlim=(0.0, 1.03 * x_max),
        ylim=(1.05 * min(y_min, -x_max), 0.08 * x_max),
    )

    ax_a_x_ticks = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    ax_a_y_ticks = np.array([-0.75, -0.50, -0.25, 0.00])
    ax_a.set_xticks(ax_a_x_ticks)
    ax_a.set_yticks(ax_a_y_ticks)

    ax_a.legend(
        loc="upper right", fontsize=LEGEND_FONTSIZE, frameon=False,
        handlelength=1.8, labelspacing=0.25, borderaxespad=0.4,
    )
    ax_a.set_aspect("equal", adjustable="box")

    markers = {8.0: "o", 10.0: "s", 12.0: "^"}
    scatter_alpha = 0.70 if FREQUENCY_MODE == "gaussian" else 1.0
    for j, K in enumerate(PANEL_B_K_VALUES):
        mask = np.isclose(plot_data["panel_b_K"], K)
        ax_b.scatter(
            plot_data["panel_b_Lambda_K"][mask],
            plot_data["panel_b_D_meas"][mask],
            marker=markers[K],
            s=58,
            color=colors[j],
            alpha=scatter_alpha,
            edgecolors="black",
            linewidths=0.7,
            label=rf"$K={K:g}$",
            zorder=3,
        )
    lo = min(float(np.min(plot_data["panel_b_Lambda_K"])), float(np.min(plot_data["panel_b_D_meas"])))
    hi = max(float(np.max(plot_data["panel_b_Lambda_K"])), float(np.max(plot_data["panel_b_D_meas"])))
    pad = 0.09 * (hi - lo)
    ax_b.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.8, zorder=2)
    ax_b.set(
        xlabel=r"$\Lambda_K$",
        ylabel=r"$D_{\mathrm{meas}}$",
        xlim=(lo - pad, hi + pad),
        ylim=(lo - pad, hi + pad),
    )

    common_ticks = np.array([2.0e-3, 3.0e-3, 4.0e-3, 5.0e-3, 6.0e-3, 7.0e-3])
    ax_b.set_xticks(common_ticks)
    ax_b.set_yticks(common_ticks)

    ax_b.legend(
        loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False,
        handlelength=1.2, labelspacing=0.35, borderaxespad=0.4,
    )
    ax_b.set_aspect("equal", adjustable="box")
    for label, ax in zip(("(a)", "(b)"), axes):
        ax.xaxis.labelpad = XLABEL_PAD
        ax.yaxis.labelpad = YLABEL_PAD
        format_axes(ax)
        ax.text(
            0.5, PANEL_LABEL_Y, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=PANEL_LABEL_FONTSIZE,
        )
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)
    saved_paths = save_figure_all_formats(fig, FIGURE_DIR, FIGURE_STEM)
    print("Saved:", *saved_paths, sep="\n")
    record_figure_regression(PROJECT_ROOT, figure_script="fig03_slow_polar_drift.py", generated_artifacts=saved_paths)
    plt.close(fig)

#%%
# Draw selected mode
plot_figure3(data)

# %%
