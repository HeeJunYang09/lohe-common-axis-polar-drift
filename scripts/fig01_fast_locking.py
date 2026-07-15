#%%
# Figure 01: fast transverse locking

#%%
# Imports and project paths
import csv
import os
import sys
from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import jax

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
    integrate_sphere_model,
    locking_diagnostics,
    make_theorem_regime_initial_condition,
    save_metadata,
    solve_locked_profile,
)
from figure_style import apply_paper_style, color_cycle, format_axes, save_figure_all_formats
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

# plt.rcParams.update({
#     "font.size": 12,
#     "axes.labelsize": 13,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 10.5,
# })

#%%
# Parameters: c_init is named perturb_scale in the numerical utilities.
RECOMPUTE = os.environ.get("FIG01_RECOMPUTE", "0") == "1"
SCHEMA_VERSION = "fig01_fast_locking_x64_v2"
PRECISION_MODE = "x64"
N = 128
K_VALUES = np.array([10.0, 15.0, 22.5, 30.0])
SIGMA_OMEGA = 0.15
OMEGA_BAR = 0.5
THETA0 = 0.3
PHI0 = 0.85
C_INIT = 0.30
C_TOL = 5.0
T1 = 1.8
NUM_SAVE = 2000
RTOL = 1.0e-9
ATOL = 1.0e-11

FIGURE_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed"
for directory in (FIGURE_DIR, CACHE_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "fig01_fast_locking.npz"

if not bool(jax.config.jax_enable_x64):
    raise RuntimeError("Figure 1 requires JAX x64 precision.")


def _cache_string(value):
    return str(np.asarray(value).item())


def _cache_float(value):
    return float(np.asarray(value).item())


def cache_matches_current_schema(cache):
    required = {
        "schema_version", "precision_mode", "jax_x64", "rtol", "atol",
        "time_dtype", "state_dtype", "ts", "K_values", "E_lock",
        "rho", "thresholds", "tf_num", "sphere_norm_error",
    }
    if not required.issubset(set(cache.files)):
        return False
    return (
        _cache_string(cache["schema_version"]) == SCHEMA_VERSION
        and _cache_string(cache["precision_mode"]) == PRECISION_MODE
        and bool(np.asarray(cache["jax_x64"]).item())
        and np.isclose(_cache_float(cache["rtol"]), RTOL, rtol=0.0, atol=0.0)
        and np.isclose(_cache_float(cache["atol"]), ATOL, rtol=0.0, atol=0.0)
        and _cache_string(cache["time_dtype"]) == "float64"
        and _cache_string(cache["state_dtype"]) == "float64"
    )

#%%
# Load or generate data. Set RECOMPUTE=True and rerun this cell to regenerate.
if CACHE_FILE.exists() and not RECOMPUTE:
    loaded = np.load(CACHE_FILE, allow_pickle=False)
    if cache_matches_current_schema(loaded):
        data = dict(loaded)
        loaded.close()
        print("Loaded:", CACHE_FILE)
    else:
        loaded.close()
        print("Ignoring stale Figure 1 cache:", CACHE_FILE)
        data = None
else:
    data = None

if data is None:
    records = []
    for K in K_VALUES:
        omega = deterministic_frequencies(N=N, omega_bar=OMEGA_BAR, sigma_omega=SIGMA_OMEGA)
        stats = frequency_statistics(omega, K)
        vartheta, locked_residual = solve_locked_profile(omega, K)
        x0, _ = make_theorem_regime_initial_condition(
            omega, K, vartheta, theta0=THETA0, phi0=PHI0, perturb_scale=C_INIT
        )
        result = integrate_sphere_model(
            omega, K, x0, t0=0.0, t1=T1, num_save=NUM_SAVE, rtol=RTOL, atol=ATOL
        )
        diag = locking_diagnostics(result.ts, result.xs, stats["omega_bar"], vartheta)
        threshold = C_TOL * stats["rho"] ** 2
        idx_tf = first_threshold_index(diag["E_lock"], threshold)
        records.append(
            {
                "ts": result.ts,
                "E_theta": diag["E_theta"],
                "E_phi": diag["E_phi"],
                "E_lock": diag["E_lock"],
                "rho": stats["rho"],
                "threshold": threshold,
                "idx_tf": -1 if idx_tf is None else idx_tf,
                "tf_num": np.nan if idx_tf is None else result.ts[idx_tf],
                "Lambda_K": compute_lambda_K(vartheta, K),
                "locked_residual": locked_residual,
                "sphere_norm_error": result.stats["sphere_norm_error"],
                "time_dtype": str(result.ts.dtype),
                "state_dtype": str(result.xs.dtype),
            }
        )
    data = {
        "schema_version": np.array(SCHEMA_VERSION),
        "precision_mode": np.array(PRECISION_MODE),
        "jax_x64": np.array(bool(jax.config.jax_enable_x64)),
        "rtol": np.array(RTOL),
        "atol": np.array(ATOL),
        "time_dtype": np.array(records[0]["time_dtype"]),
        "state_dtype": np.array(records[0]["state_dtype"]),
        "ts": records[0]["ts"],
        "K_values": K_VALUES,
        "E_theta": np.stack([r["E_theta"] for r in records]),
        "E_phi": np.stack([r["E_phi"] for r in records]),
        "E_lock": np.stack([r["E_lock"] for r in records]),
        "rho": np.array([r["rho"] for r in records]),
        "thresholds": np.array([r["threshold"] for r in records]),
        "idx_tf": np.array([r["idx_tf"] for r in records], dtype=int),
        "tf_num": np.array([r["tf_num"] for r in records]),
        "Lambda_K": np.array([r["Lambda_K"] for r in records]),
        "locked_residual": np.array([r["locked_residual"] for r in records]),
        "sphere_norm_error": np.array([r["sphere_norm_error"] for r in records]),
    }
    np.savez_compressed(CACHE_FILE, **data)
    print("Saved:", CACHE_FILE)

#%%
# Diagnostics and summary files
summary_rows = []
table_rows = []
for j, K in enumerate(data["K_values"]):
    rho = float(data["rho"][j])
    idx = int(data["idx_tf"][j])
    tf = float(data["tf_num"][j])
    crossed = idx >= 0 and np.isfinite(tf)
    row = {
        "K": K,
        "rho": rho,
        "threshold": data["thresholds"][j],
        "t_f_num": tf,
        "t_f_index": idx,
        "K_t_f_num": K * tf,
        "K_t_f_num_over_log_one_over_rho": K * tf / np.log(1.0 / rho),
        "E_theta_initial": data["E_theta"][j, 0],
        "E_phi_initial": data["E_phi"][j, 0],
        "E_lock_initial": data["E_lock"][j, 0],
        "E_theta_initial_over_rho": data["E_theta"][j, 0] / rho,
        "E_phi_initial_over_rho": data["E_phi"][j, 0] / rho,
        "E_lock_initial_over_rho": data["E_lock"][j, 0] / rho,
        "E_theta_at_t_f_num": data["E_theta"][j, idx] if crossed else np.nan,
        "E_phi_at_t_f_num": data["E_phi"][j, idx] if crossed else np.nan,
        "E_lock_at_t_f_num": data["E_lock"][j, idx] if crossed else np.nan,
        "E_theta_at_t_f_num_over_rho2": data["E_theta"][j, idx] / rho**2 if crossed else np.nan,
        "E_phi_at_t_f_num_over_rho2": data["E_phi"][j, idx] / rho**2 if crossed else np.nan,
        "E_lock_at_t_f_num_over_rho2": data["E_lock"][j, idx] / rho**2 if crossed else np.nan,
        "threshold_crossed": crossed,
        "locked_profile_residual": data["locked_residual"][j],
        "sphere_norm_error": data["sphere_norm_error"][j],
    }
    summary_rows.append(row)
    table_rows.append({key: row[key] for key in (
        "K", "rho", "t_f_num", "K_t_f_num", "K_t_f_num_over_log_one_over_rho",
        "E_theta_at_t_f_num_over_rho2", "E_phi_at_t_f_num_over_rho2"
    )})

for filename, rows in (
    ("summary_exp01_fast_locking.csv", summary_rows),
    ("fast_layer_diagnostic_table.csv", table_rows),
):
    with (SUMMARY_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save_metadata(SUMMARY_DIR / "metadata_exp01.json", {
    "figure": "Figure 1", "cache_file": CACHE_FILE.relative_to(PROJECT_ROOT), "N": N, "K_values": K_VALUES,
    "sigma_omega": SIGMA_OMEGA, "theta0": THETA0, "phi0": PHI0,
    "c_init": C_INIT, "C_tol": C_TOL, "rtol": RTOL, "atol": ATOL,
    "schema_version": SCHEMA_VERSION, "precision_mode": PRECISION_MODE,
    "jax_x64": bool(jax.config.jax_enable_x64),
    "time_dtype": _cache_string(data["time_dtype"]), "state_dtype": _cache_string(data["state_dtype"]),
})

#%%
# Plot: layout and labels are intentionally easy to edit here.
ts = data["ts"]
normalized = data["E_lock"] / data["thresholds"][:, None]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.64))

# FIGSIZE = (7.0, 3.55)
LEGEND_Y = 0.925
LEGEND_FONTSIZE = 13

FIGSIZE = (7.8, 4.5)
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.095, 0.985, 0.28, 0.94, 0.34
PANEL_LABEL_Y = -0.25
PANEL_LABEL_FONTSIZE = 17
XLABEL_PAD = 4
YLABEL_PAD = 4

XLABEL_PAD = 4
YLABEL_PAD = 4

for j, K in enumerate(data["K_values"]):
    color = color_cycle()[j]
    axes[0].semilogy(ts, data["E_lock"][j], color=color, label=rf"$K={K:g}$")
    axes[1].semilogy(K * ts, normalized[j], color=color, label=rf"$K={K:g}$")
    idx = int(data["idx_tf"][j])
    if idx >= 0:
        axes[1].plot(K * ts[idx], normalized[j, idx], "o", color=color,
                     markeredgecolor="black", markeredgewidth=0.7, zorder=5)
axes[1].axhline(1.0, color="0.25", linestyle="--", linewidth=1.4)
# axes[0].set(xlabel=r"$t$", ylabel=r"$E_{\mathrm{lock}}(t)$", xlim=(0.0, 0.4))
# axes[1].set(xlabel=r"$Kt$", ylabel=r"$E_{\mathrm{lock}}/(C_{\mathrm{tol}}\rho^2)$",
#             xlim=(0.0, 10.0))

axes[0].set(
    xlabel=r"$t$",
    ylabel=r"$E_{\mathrm{lock}}(t)$",
    xlim=(0.0, 0.4),
)

axes[1].set(
    xlabel=r"$Kt$",
    ylabel=r"Normalized error",
    xlim=(0.0, 10.0),
)

axes[0].xaxis.labelpad = XLABEL_PAD
axes[0].yaxis.labelpad = YLABEL_PAD
axes[1].xaxis.labelpad = XLABEL_PAD
axes[1].yaxis.labelpad = YLABEL_PAD

for ax in axes:
    format_axes(ax)
# axes[0].text(0.5, -0.30, r"(a) early physical time", transform=axes[0].transAxes,
#              ha="center", va="top", fontsize=14)
# axes[1].text(0.5, -0.30, r"(b) normalized rescaled time", transform=axes[1].transAxes,
#              ha="center", va="top", fontsize=14)

axes[0].text(
    0.5, PANEL_LABEL_Y, r"(a)",
    transform=axes[0].transAxes,
    ha="center", va="top",
    fontsize=PANEL_LABEL_FONTSIZE,
)

axes[1].text(
    0.5, PANEL_LABEL_Y, r"(b)",
    transform=axes[1].transAxes,
    ha="center", va="top",
    fontsize=PANEL_LABEL_FONTSIZE,
)

handles, labels = axes[1].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))
# fig.subplots_adjust(left=0.09, right=0.99, bottom=0.30, top=0.82, wspace=0.34)

# fig.legend(
#     handles,
#     labels,
#     loc="upper center",
#     ncol=4,
#     bbox_to_anchor=(0.5, 0.92),
#     frameon=False,
#     handlelength=2.0,
#     columnspacing=1.1,
#     fontsize=10.5,
# )

axes[1].legend(
    handles,
    labels,
    loc="upper right",
    ncol=2,
    frameon=True,
    framealpha=0.85,
    facecolor="white",
    edgecolor="none",
    fontsize=LEGEND_FONTSIZE,
    handlelength=1.7,
    columnspacing=0.9,
    labelspacing=0.25,
    borderpad=0.35,
)

fig.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    bottom=BOTTOM,
    top=TOP,
    wspace=WSPACE,
)


#%%
# Save and show
saved_paths = save_figure_all_formats(fig, FIGURE_DIR, "fig1_fast_locking")
print("Saved:", *saved_paths, sep="\n")
record_figure_regression(PROJECT_ROOT, figure_script="fig01_fast_locking.py", generated_artifacts=saved_paths)
plt.close(fig)

# %%
