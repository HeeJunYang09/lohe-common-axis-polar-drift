#%%
# Figure 04: hitting-time law and random-frequency robustness

#%%
# Imports and project paths
import csv
import sys
from pathlib import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
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
    frequency_statistics,
    gaussian_frequencies,
    integrate_sphere_model,
    interpolated_hitting_time,
    locking_diagnostics,
    make_theorem_regime_initial_condition,
    save_metadata,
    solve_locked_profile,
)
from figure_style import apply_paper_style, format_axes, save_figure_all_formats

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
# Parameters. Change only FREQUENCY_MODE to switch between the two figures.
RECOMPUTE = False
FREQUENCY_MODE = "deterministic"  # "deterministic" or "gaussian"
# FREQUENCY_MODE = "gaussian"  # "deterministic" or "gaussian"

N = 28
K_VALUES = np.array([6.0, 7.5, 9.0, 10.5, 12.0])
SIGMA_VALUES = np.array([0.20, 0.24, 0.28, 0.32, 0.36])
SEEDS = np.arange(5)
ETA = 0.82
OMEGA_BAR = 0.5
THETA0 = 0.3
PHI0 = 0.85
C_INIT = 0.30
C_TOL = 5.0
T1 = 300.0
NUM_SAVE = 2200
RTOL = 1.0e-7
ATOL = 1.0e-9

FIGURE_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed"
for directory in (FIGURE_DIR, CACHE_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GRID_TAG = f"{len(K_VALUES)}x{len(SIGMA_VALUES)}"
SEED_TAG = f"seeds{int(SEEDS[0])}-{int(SEEDS[-1])}"

if FREQUENCY_MODE == "deterministic":
    CACHE_FILE = CACHE_DIR / f"fig04_hitting_time_deterministic_{GRID_TAG}.npz"
    SUMMARY_FILE = SUMMARY_DIR / f"summary_exp04_hitting_time_deterministic_{GRID_TAG}.csv"
    METADATA_FILE = SUMMARY_DIR / f"metadata_exp04_deterministic_{GRID_TAG}.json"
    FIGURE_STEM = "fig4_hitting_time_deterministic"
elif FREQUENCY_MODE == "gaussian":
    CACHE_FILE = CACHE_DIR / f"fig04_hitting_time_gaussian_{GRID_TAG}_{SEED_TAG}.npz"
    SUMMARY_FILE = SUMMARY_DIR / f"summary_exp04_hitting_time_gaussian_{GRID_TAG}_{SEED_TAG}.csv"
    SEED_METRICS_FILE = (
        SUMMARY_DIR / f"summary_exp04_hitting_time_gaussian_seed_metrics_{GRID_TAG}_{SEED_TAG}.csv"
    )
    METADATA_FILE = SUMMARY_DIR / f"metadata_exp04_gaussian_{GRID_TAG}_{SEED_TAG}.json"
    FIGURE_STEM = "fig4_hitting_time_gaussian"
    ROBUSTNESS_FIGURE_STEM = "fig4_hitting_time_gaussian_robustness"
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
            N=N,
            omega_bar=OMEGA_BAR,
            sigma_omega=sigma,
            seed=int(seed),
        )
    raise ValueError(f"Unknown FREQUENCY_MODE: {FREQUENCY_MODE}")


def simulate_case(K, sigma, seed=None):
    omega = make_frequencies(sigma=sigma, seed=seed)
    stats = frequency_statistics(omega, K)
    vartheta, locked_residual = solve_locked_profile(omega, K)
    Lambda_K = compute_lambda_K(vartheta, K)
    x0, _ = make_theorem_regime_initial_condition(
        omega,
        K,
        vartheta,
        theta0=THETA0,
        phi0=PHI0,
        perturb_scale=C_INIT,
    )
    result = integrate_sphere_model(
        omega,
        K,
        x0,
        t0=0.0,
        t1=T1,
        num_save=NUM_SAVE,
        rtol=RTOL,
        atol=ATOL,
    )
    diag = locking_diagnostics(result.ts, result.xs, stats["omega_bar"], vartheta)

    fast_threshold = C_TOL * stats["rho"] ** 2
    idx = first_threshold_index(diag["E_lock"], fast_threshold)
    fast_threshold_reached = idx is not None
    if idx is None:
        idx = int(np.argmin(diag["E_lock"]))

    tf_num = float(result.ts[idx])
    phi_tf = float(diag["phi_bar"][idx])
    T_sim = interpolated_hitting_time(result.ts, diag["phi_bar"], ETA, start_time=tf_num)
    hitting_threshold_reached = bool(np.isfinite(T_sim))

    if Lambda_K > 0.0 and phi_tf > ETA:
        T_pred = tf_num + np.log(np.tan(phi_tf) / np.tan(ETA)) / Lambda_K
    elif phi_tf <= ETA:
        T_pred = tf_num
    else:
        T_pred = float("nan")

    relative_error = (
        abs(T_sim - T_pred) / abs(T_sim)
        if np.isfinite(T_sim) and np.isfinite(T_pred) and T_sim != 0.0
        else float("nan")
    )
    return {
        "seed": -1 if seed is None else int(seed),
        "K": float(K),
        "sigma": float(sigma),
        "rho": float(stats["rho"]),
        "max_delta": float(stats["max_delta"]),
        "Lambda_K": float(Lambda_K),
        "tf_num": tf_num,
        "phi_tf": phi_tf,
        "T_sim": float(T_sim),
        "T_pred": float(T_pred),
        "relative_error": float(relative_error),
        "locked_residual": float(locked_residual),
        "sphere_norm_error": float(result.stats["sphere_norm_error"]),
        "fast_threshold_reached": fast_threshold_reached,
        "hitting_threshold_reached": hitting_threshold_reached,
    }

#%%
# Load or generate mode-specific data. RECOMPUTE=True regenerates this mode only.
if CACHE_FILE.exists() and not RECOMPUTE:
    data = dict(np.load(CACHE_FILE))
    print("Loaded:", CACHE_FILE)
else:
    if FREQUENCY_MODE == "deterministic":
        records = [
            simulate_case(K, sigma)
            for K in K_VALUES
            for sigma in SIGMA_VALUES
        ]
        T_grid = np.asarray([r["T_sim"] for r in records]).reshape(
            len(K_VALUES), len(SIGMA_VALUES)
        )
        data = {
            "frequency_mode": np.array(FREQUENCY_MODE),
            "K_values": K_VALUES,
            "sigma_values": SIGMA_VALUES,
            "T_grid": T_grid,
        }
    else:
        records = [
            simulate_case(K, sigma, seed=seed)
            for seed in SEEDS
            for K in K_VALUES
            for sigma in SIGMA_VALUES
        ]
        data = {
            "frequency_mode": np.array(FREQUENCY_MODE),
            "seeds": SEEDS,
            "K_values": K_VALUES,
            "sigma_values": SIGMA_VALUES,
            "seed_case": np.array([r["seed"] for r in records], dtype=int),
        }

    data.update(
        {
            "K_case": np.array([r["K"] for r in records]),
            "sigma_case": np.array([r["sigma"] for r in records]),
            "rho": np.array([r["rho"] for r in records]),
            "max_delta": np.array([r["max_delta"] for r in records]),
            "Lambda_K": np.array([r["Lambda_K"] for r in records]),
            "tf_num": np.array([r["tf_num"] for r in records]),
            "phi_tf": np.array([r["phi_tf"] for r in records]),
            "T_sim": np.array([r["T_sim"] for r in records]),
            "T_pred": np.array([r["T_pred"] for r in records]),
            "relative_error": np.array([r["relative_error"] for r in records]),
            "locked_residual": np.array([r["locked_residual"] for r in records]),
            "sphere_norm_error": np.array([r["sphere_norm_error"] for r in records]),
            "fast_threshold_reached": np.array(
                [r["fast_threshold_reached"] for r in records], dtype=bool
            ),
            "hitting_threshold_reached": np.array(
                [r["hitting_threshold_reached"] for r in records], dtype=bool
            ),
        }
    )
    np.savez_compressed(CACHE_FILE, **data)
    print("Saved:", CACHE_FILE)

#%%
# Diagnostics, summaries, and mode-specific metadata
def error_metrics(T_sim, T_pred, relative_error):
    valid = np.isfinite(T_sim) & np.isfinite(T_pred)
    finite_relative = relative_error[np.isfinite(relative_error)]
    if not np.any(valid):
        return {
            "relative_L2_error": float("nan"),
            "max_relative_error": float("nan"),
            "mean_relative_error": float("nan"),
        }
    return {
        "relative_L2_error": float(
            np.linalg.norm(T_sim[valid] - T_pred[valid]) / np.linalg.norm(T_sim[valid])
        ),
        "max_relative_error": float(np.max(finite_relative)),
        "mean_relative_error": float(np.mean(finite_relative)),
    }


global_metrics = error_metrics(data["T_sim"], data["T_pred"], data["relative_error"])
global_metrics.update(
    {
        "rho_min": float(np.min(data["rho"])),
        "rho_max": float(np.max(data["rho"])),
        "all_fast_thresholds_reached": bool(np.all(data["fast_threshold_reached"])),
        "all_hitting_thresholds_reached": bool(np.all(data["hitting_threshold_reached"])),
    }
)

rows = []
for idx in range(len(data["T_sim"])):
    rows.append(
        {
            "frequency_mode": FREQUENCY_MODE,
            "seed": int(data["seed_case"][idx]) if "seed_case" in data else -1,
            "K": data["K_case"][idx],
            "sigma_omega": data["sigma_case"][idx],
            "rho": data["rho"][idx],
            "max_delta": data["max_delta"][idx],
            "Lambda_K": data["Lambda_K"][idx],
            "eta": ETA,
            "t_f_num": data["tf_num"][idx],
            "phi_bar_tf": data["phi_tf"][idx],
            "T_eta_pred": data["T_pred"][idx],
            "T_eta_sim": data["T_sim"][idx],
            "relative_error": data["relative_error"][idx],
            "fast_threshold_reached": bool(data["fast_threshold_reached"][idx]),
            "hitting_threshold_reached": bool(data["hitting_threshold_reached"][idx]),
            "locked_profile_residual": data["locked_residual"][idx],
            "sphere_norm_error": data["sphere_norm_error"][idx],
        }
    )

with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

seed_metric_rows = []
if FREQUENCY_MODE == "gaussian":
    for seed in SEEDS:
        mask = data["seed_case"] == seed
        metrics = error_metrics(
            data["T_sim"][mask], data["T_pred"][mask], data["relative_error"][mask]
        )
        seed_metric_rows.append(
            {
                "seed": int(seed),
                **metrics,
                "rho_min_seed": float(np.min(data["rho"][mask])),
                "rho_max_seed": float(np.max(data["rho"][mask])),
                "all_fast_thresholds_reached_seed": bool(
                    np.all(data["fast_threshold_reached"][mask])
                ),
                "all_hitting_thresholds_reached_seed": bool(
                    np.all(data["hitting_threshold_reached"][mask])
                ),
            }
        )
    with SEED_METRICS_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_metric_rows[0]))
        writer.writeheader()
        writer.writerows(seed_metric_rows)

metadata = {
    "figure": "Figure 4",
    "frequency_mode": FREQUENCY_MODE,
    "grid_tag": GRID_TAG,
    "cache_file": CACHE_FILE.relative_to(PROJECT_ROOT),
    "summary_file": SUMMARY_FILE.relative_to(PROJECT_ROOT),
    "metadata_file": METADATA_FILE.relative_to(PROJECT_ROOT),
    "figure_stem": FIGURE_STEM,
    "N": N,
    "K_values": K_VALUES,
    "sigma_values": SIGMA_VALUES,
    "eta": ETA,
    "theta0": THETA0,
    "phi0": PHI0,
    "c_init": C_INIT,
    "C_tol": C_TOL,
    **global_metrics,
}
if FREQUENCY_MODE == "gaussian":
    metadata.update(
        {
            "seeds": SEEDS,
            "seed_tag": SEED_TAG,
            "seed_metrics_file": SEED_METRICS_FILE.relative_to(PROJECT_ROOT),
            "seed_metrics": seed_metric_rows,
        }
    )
save_metadata(METADATA_FILE, metadata)

print("Saved:", SUMMARY_FILE)
print("Saved:", METADATA_FILE)
if FREQUENCY_MODE == "gaussian":
    print("Saved:", SEED_METRICS_FILE)
print("Global metrics:", global_metrics)

#%%
# Plotting helpers
FIGSIZE = (7.8, 4.5)
WIDTH_RATIOS = [1.05, 1.15]
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.09, 0.955, 0.28, 0.93, 0.34
PANEL_LABEL_Y = -0.25
PANEL_LABEL_FONTSIZE = 17
HEATMAP_TEXT_FONTSIZE = 12

def grid_edges(values):
    mids = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        [[values[0] - 0.5 * (values[1] - values[0])],
         mids,
         [values[-1] + 0.5 * (values[-1] - values[-2])]]
    )


def add_cell_labels(ax, mesh, values, x_values, y_values, append_percent=False):
    for i, y_value in enumerate(y_values):
        for j, x_value in enumerate(x_values):
            value = values[i, j]
            if not np.isfinite(value):
                text = "--"
                color = "black"
            else:
                text = f"{value:.1f}%" if append_percent else f"{value:.1f}"
                color = "white" if mesh.norm(value) < 0.55 else "black"
            ax.text(
                x_value,
                y_value,
                text,
                ha="center",
                va="center",
                fontsize=HEATMAP_TEXT_FONTSIZE,
                color=color,
            )


def finish_figure(fig, axes, figure_stem, wspace=WSPACE):
    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(
            0.5,
            PANEL_LABEL_Y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
        )
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=wspace)
    saved_paths = save_figure_all_formats(fig, FIGURE_DIR, figure_stem)
    print("Saved:", *saved_paths, sep="\n")
    plt.close(fig)


def plot_deterministic_figure4(plot_data):
    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": WIDTH_RATIOS}
    )
    ax_a, ax_b = axes
    valid = np.isfinite(plot_data["T_pred"]) & np.isfinite(plot_data["T_sim"])
    ax_a.scatter(
        plot_data["T_pred"][valid],
        plot_data["T_sim"][valid],
        color="#0072B2",
        s=48,
        edgecolors="black",
        linewidths=0.7,
    )
    lo = min(float(np.min(plot_data["T_pred"][valid])), float(np.min(plot_data["T_sim"][valid])))
    hi = max(float(np.max(plot_data["T_pred"][valid])), float(np.max(plot_data["T_sim"][valid])))
    pad = 0.08 * (hi - lo)
    ax_a.plot([0, 20], [0, 20], "k--", linewidth=1.8)
    ax_a.set(
        xlabel=r"$T_\eta^{\mathrm{pred}}$",
        ylabel=r"$T_\eta^{\mathrm{sim}}$",
        xlim=(0, 20),
        ylim=(0, 20),
    )
    ax_a.set_xticks([0, 5, 10, 15, 20])
    ax_a.set_yticks([5, 10, 15, 20])
    ax_a.set_aspect("equal", adjustable="box")
    format_axes(ax_a)

    mesh = ax_b.pcolormesh(
        grid_edges(plot_data["sigma_values"]),
        grid_edges(plot_data["K_values"]),
        np.ma.masked_invalid(plot_data["T_grid"]),
        shading="flat",
        cmap="viridis",
        edgecolors="white",
        linewidth=0.8,
    )
    ax_b.set(xlabel=r"$\sigma_\omega$", ylabel=r"$K$")
    ax_b.set_xticks(plot_data["sigma_values"])
    ax_b.set_yticks(plot_data["K_values"])
    add_cell_labels(
        ax_b, mesh, plot_data["T_grid"], plot_data["sigma_values"], plot_data["K_values"]
    )
    divider = make_axes_locatable(ax_b)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$T_\eta^{\mathrm{sim}}$", labelpad=5)

    heatmap_values = np.asarray(plot_data["T_grid"], dtype=float)
    vmin = float(np.nanmin(heatmap_values))
    vmax = float(np.nanmax(heatmap_values))

    cbar_ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=14)

    finish_figure(fig, axes, FIGURE_STEM)


def aggregate_gaussian_grids(plot_data):
    K_values = np.asarray(plot_data["K_values"], dtype=float)
    sigma_values = np.asarray(plot_data["sigma_values"], dtype=float)
    T_pred_mean = np.full((len(K_values), len(sigma_values)), np.nan)
    T_sim_mean = np.full((len(K_values), len(sigma_values)), np.nan)
    T_sim_std = np.full_like(T_sim_mean, np.nan)
    rel_mean = np.full_like(T_sim_mean, np.nan)
    rel_max = np.full_like(T_sim_mean, np.nan)
    rho_mean = np.full_like(T_sim_mean, np.nan)
    rho_max = np.full_like(T_sim_mean, np.nan)

    for i, K in enumerate(K_values):
        for j, sigma in enumerate(sigma_values):
            mask = np.isclose(plot_data["K_case"], K) & np.isclose(
                plot_data["sigma_case"], sigma
            )
            T_pred_vals = np.asarray(plot_data["T_pred"][mask], dtype=float)
            T_vals = np.asarray(plot_data["T_sim"][mask], dtype=float)
            rel_vals = np.asarray(plot_data["relative_error"][mask], dtype=float)
            rho_vals = np.asarray(plot_data["rho"][mask], dtype=float)
            T_pred_mean[i, j] = np.nanmean(T_pred_vals)
            T_sim_mean[i, j] = np.nanmean(T_vals)
            T_sim_std[i, j] = np.nanstd(T_vals)
            rel_mean[i, j] = np.nanmean(rel_vals)
            rel_max[i, j] = np.nanmax(rel_vals)
            rho_mean[i, j] = np.nanmean(rho_vals)
            rho_max[i, j] = np.nanmax(rho_vals)

    return {
        "T_pred_mean": T_pred_mean,
        "T_sim_mean": T_sim_mean,
        "T_sim_std": T_sim_std,
        "rel_mean": rel_mean,
        "rel_max": rel_max,
        "rho_mean": rho_mean,
        "rho_max": rho_max,
    }


def plot_gaussian_same_format_figure4(plot_data):
    grids = aggregate_gaussian_grids(plot_data)
    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": WIDTH_RATIOS}
    )
    ax_a, ax_b = axes
    valid = np.isfinite(plot_data["T_pred"]) & np.isfinite(plot_data["T_sim"])
    ax_a.scatter(
        plot_data["T_pred"][valid],
        plot_data["T_sim"][valid],
        color="#0072B2",
        s=48,
        edgecolors="black",
        linewidths=0.7,
    )
    lo = min(float(np.min(plot_data["T_pred"][valid])), float(np.min(plot_data["T_sim"][valid])))
    hi = max(float(np.max(plot_data["T_pred"][valid])), float(np.max(plot_data["T_sim"][valid])))
    pad = 0.08 * (hi - lo)
    ax_a.plot([0, 20], [0, 20], "k--", linewidth=1.8)
    ax_a.set(
        xlabel=r"$T_\eta^{\mathrm{pred}}$",
        ylabel=r"$T_\eta^{\mathrm{sim}}$",
        xlim=(0, 20),
        ylim=(0, 20),
    )
    ax_a.set_xticks([0, 5, 10, 15, 20])
    ax_a.set_yticks([5, 10, 15, 20])
    ax_a.set_aspect("equal", adjustable="box")
    format_axes(ax_a)

    mesh = ax_b.pcolormesh(
        grid_edges(plot_data["sigma_values"]),
        grid_edges(plot_data["K_values"]),
        np.ma.masked_invalid(grids["T_sim_mean"]),
        shading="flat",
        cmap="viridis",
        edgecolors="white",
        linewidth=0.8,
    )
    ax_b.set(xlabel=r"$\sigma_\omega$", ylabel=r"$K$")
    ax_b.set_xticks(plot_data["sigma_values"])
    ax_b.set_yticks(plot_data["K_values"])
    ax_b.set_box_aspect(1)
    add_cell_labels(
        ax_b,
        mesh,
        grids["T_sim_mean"],
        plot_data["sigma_values"],
        plot_data["K_values"],
    )
    divider = make_axes_locatable(ax_b)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$T_\eta^{\mathrm{sim}}$", labelpad=5)

    heatmap_values = np.asarray(grids["T_sim_mean"], dtype=float)
    vmin = float(np.nanmin(heatmap_values))
    vmax = float(np.nanmax(heatmap_values))

    cbar_ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=14)

    finish_figure(fig, axes, FIGURE_STEM)


def plot_gaussian_robustness_figure4(plot_data):
    grids = aggregate_gaussian_grids(plot_data)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, FIGSIZE[1]), gridspec_kw={"width_ratios": WIDTH_RATIOS}
    )
    ax_a, ax_b = axes
    valid = np.isfinite(plot_data["T_pred"]) & np.isfinite(plot_data["T_sim"])
    scatter = ax_a.scatter(
        plot_data["T_pred"][valid],
        plot_data["T_sim"][valid],
        c=plot_data["rho"][valid],
        cmap="viridis",
        s=25,
        edgecolors="black",
        linewidths=0.35,
    )
    lo = min(float(np.min(plot_data["T_pred"][valid])), float(np.min(plot_data["T_sim"][valid])))
    hi = max(float(np.max(plot_data["T_pred"][valid])), float(np.max(plot_data["T_sim"][valid])))
    pad = 0.08 * (hi - lo)
    ax_a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.8)
    ax_a.set(
        xlabel=r"$T_\eta^{\mathrm{pred}}$",
        ylabel=r"$T_\eta^{\mathrm{sim}}$",
        xlim=(lo - pad, hi + pad),
        ylim=(lo - pad, hi + pad),
    )
    ax_a.set_aspect("equal", adjustable="box")
    format_axes(ax_a)

    divider = make_axes_locatable(ax_a)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    cbar_a = fig.colorbar(scatter, cax=cax)
    cbar_a.set_label(r"$\rho$", labelpad=2)
    cbar_a.ax.tick_params(labelsize=11)

    rel_mean_percent = 100.0 * grids["rel_mean"]
    mesh = ax_b.pcolormesh(
        grid_edges(plot_data["sigma_values"]),
        grid_edges(plot_data["K_values"]),
        np.ma.masked_invalid(rel_mean_percent),
        shading="flat",
        cmap="magma_r",
        edgecolors="white",
        linewidth=0.8,
    )
    ax_b.set(xlabel=r"$\sigma_\omega$", ylabel=r"$K$")
    ax_b.set_xticks(plot_data["sigma_values"])
    ax_b.set_yticks(plot_data["K_values"])
    ax_b.set_box_aspect(1)
    add_cell_labels(
        ax_b,
        mesh,
        rel_mean_percent,
        plot_data["sigma_values"],
        plot_data["K_values"],
        append_percent=True,
    )
    
    divider = make_axes_locatable(ax_b)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    cbar_b = fig.colorbar(mesh, cax=cax)
    cbar_b.set_label(r"Mean relative error (\%)", labelpad=5)
    cbar_b.ax.tick_params(labelsize=14)
    
    finish_figure(fig, axes, ROBUSTNESS_FIGURE_STEM, wspace=0.52)

#%%
# Draw the selected mode
if FREQUENCY_MODE == "deterministic":
    plot_deterministic_figure4(data)
elif FREQUENCY_MODE == "gaussian":
    plot_gaussian_same_format_figure4(data)
    plot_gaussian_robustness_figure4(data)

# %%
