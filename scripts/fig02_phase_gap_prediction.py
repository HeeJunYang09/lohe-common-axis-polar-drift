#%%
# Figure 02: locked phase-gap prediction and rho^2 residual scaling

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
    deterministic_frequencies,
    first_threshold_index,
    frequency_statistics,
    integrate_sphere_model,
    locking_diagnostics,
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
# plt.rcParams.update({
#     "axes.labelsize": 13,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 10.5,
#     "font.size": 12,
# })

#%%
# Parameters and one-stop layout controls
RECOMPUTE = os.environ.get("FIG02_RECOMPUTE", "0") == "1"
SCHEMA_VERSION = "fig02_phase_gap_prediction_obs_elock_x64_v3"
PRECISION_MODE = "x64"
OBSERVATION_FACTOR = 2.0
OBSERVATION_RULE = "t_obs = observation_factor * log(1/rho) / K"
N = 48
K_REP = 12.0
K_VALUES = np.array([8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0])
SIGMA_OMEGA = 0.15
OMEGA_BAR = 0.5
THETA0 = 0.3
PHI0 = 0.85
C_INIT = 0.30
C_TOL = 5.0
T1 = 1.2
NUM_SAVE = 700
RTOL = 1.0e-9
ATOL = 1.0e-11

# FIGSIZE = (8.0, 5.6)
FIGSIZE = (7.8, 7.0)

# LEFT, RIGHT, BOTTOM, TOP = 0.14, 0.96, 0.23, 0.94
LEFT, RIGHT, BOTTOM, TOP, WSPACE = 0.095, 0.985, 0.28, 0.94, -0.55

WSPACE, HSPACE = 0.12, 0.55
HEIGHT_RATIOS = [1.0, 0.72]

XLABEL_PAD = 4

PANEL_LABEL_Y_TOP = -0.30
PANEL_LABEL_Y_BOTTOM = -0.40
PANEL_LABEL_FONTSIZE = 17

FIGURE_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed"
for directory in (FIGURE_DIR, CACHE_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "fig02_phase_gap_prediction.npz"

if not bool(jax.config.jax_enable_x64):
    raise RuntimeError("Figure 2 requires JAX x64 precision.")


def _as_cache_string(value):
    return str(np.asarray(value).item())


def _cache_float(value):
    return float(np.asarray(value).item())


def _cache_bool(value):
    return bool(np.asarray(value).item())

#%%
# Load or generate representative and scaling-sweep data.
def simulate(K):
    omega = deterministic_frequencies(N=N, omega_bar=OMEGA_BAR, sigma_omega=SIGMA_OMEGA)
    stats = frequency_statistics(omega, K)
    delta = omega - stats["omega_bar"]
    vartheta, locked_residual = solve_locked_profile(omega, K)
    x0, _ = make_theorem_regime_initial_condition(
        omega, K, vartheta, theta0=THETA0, phi0=PHI0, perturb_scale=C_INIT
    )
    t_obs = float(OBSERVATION_FACTOR * np.log(1.0 / stats["rho"]) / K)
    result = integrate_sphere_model(
        omega, K, x0, t0=0.0, t1=t_obs, num_save=NUM_SAVE, rtol=RTOL, atol=ATOL
    )
    diag = locking_diagnostics(result.ts, result.xs, stats["omega_bar"], vartheta)
    threshold = C_TOL * stats["rho"] ** 2
    idx = first_threshold_index(diag["E_lock"], threshold)
    if idx is None:
        raise RuntimeError(f"K={K:g}: fast threshold was not reached before t_obs.")
    tf_num = float(result.ts[idx])
    if t_obs <= tf_num:
        raise RuntimeError(f"K={K:g}: prescribed observation time is not post-fast.")
    residual_obs = diag["a"][-1] - vartheta
    E_theta_obs = float(diag["E_theta"][-1])
    E_phi_obs = float(diag["E_phi"][-1])
    E_lock_obs = float(diag["E_lock"][-1])
    return {
        "a_obs": diag["a"][-1], "vartheta": vartheta, "residual_obs": residual_obs,
        "delta": delta, "rho": stats["rho"], "tf_num": tf_num,
        "t_obs": t_obs, "K_t_f_num": float(K * tf_num), "K_t_obs": float(K * t_obs),
        "R_gap_obs": np.max(np.abs(residual_obs)),
        "E_theta_obs": E_theta_obs, "E_phi_obs": E_phi_obs,
        "E_lock_obs": E_lock_obs, "threshold": threshold,
        "locked_residual": locked_residual,
        "sphere_norm_error": result.stats["sphere_norm_error"],
        "time_dtype": str(result.ts.dtype),
        "state_dtype": str(result.xs.dtype),
    }

def cache_matches_current_schema(cache):
    required = {
        "schema_version", "precision_mode", "jax_x64", "rtol", "atol",
        "time_dtype", "state_dtype", "observation_rule", "observation_factor", "K_values",
        "a_obs", "vartheta", "residual_obs", "rho_values", "R_gap_obs_values",
        "E_theta_obs_values", "E_phi_obs_values", "E_lock_obs_values",
        "tf_num_values", "t_obs_values", "K_t_f_num_values", "K_t_obs_values",
        "R_gap_obs_over_rho2_values", "E_lock_obs_over_rho2_values",
        "locked_residual_values", "sphere_norm_error_values",
    }
    if not required.issubset(set(cache.files)):
        return False
    schema = _as_cache_string(cache["schema_version"])
    precision = _as_cache_string(cache["precision_mode"])
    rule = _as_cache_string(cache["observation_rule"])
    factor = _cache_float(cache["observation_factor"])
    return (
        schema == SCHEMA_VERSION
        and precision == PRECISION_MODE
        and _cache_bool(cache["jax_x64"])
        and np.isclose(_cache_float(cache["rtol"]), RTOL, rtol=0.0, atol=0.0)
        and np.isclose(_cache_float(cache["atol"]), ATOL, rtol=0.0, atol=0.0)
        and _as_cache_string(cache["time_dtype"]) == "float64"
        and _as_cache_string(cache["state_dtype"]) == "float64"
        and rule == OBSERVATION_RULE
        and np.isclose(factor, OBSERVATION_FACTOR, rtol=0.0, atol=0.0)
    )

if CACHE_FILE.exists() and not RECOMPUTE:
    loaded = np.load(CACHE_FILE, allow_pickle=False)
    if cache_matches_current_schema(loaded):
        data = dict(loaded)
        loaded.close()
        print("Loaded:", CACHE_FILE)
    else:
        loaded.close()
        print("Ignoring stale Figure 2 cache:", CACHE_FILE)
        data = None
else:
    data = None

if data is None:
    representative = simulate(K_REP)
    sweep = [simulate(K) for K in K_VALUES]
    t_obs_values = np.array([r["t_obs"] for r in sweep])
    tf_num_values = np.array([r["tf_num"] for r in sweep])
    if np.any(t_obs_values <= tf_num_values):
        raise RuntimeError("prescribed observation time is not post-fast for every K.")
    data = {
        "schema_version": np.array(SCHEMA_VERSION),
        "precision_mode": np.array(PRECISION_MODE),
        "jax_x64": np.array(bool(jax.config.jax_enable_x64)),
        "rtol": np.array(RTOL),
        "atol": np.array(ATOL),
        "time_dtype": np.array(representative["time_dtype"]),
        "state_dtype": np.array(representative["state_dtype"]),
        "observation_rule": np.array(OBSERVATION_RULE),
        "observation_factor": np.array(OBSERVATION_FACTOR),
        "C_tol": np.array(C_TOL),
        "K_values": K_VALUES,
        "a_obs": representative["a_obs"],
        "vartheta": representative["vartheta"],
        "residual_obs": representative["residual_obs"],
        "delta": representative["delta"],
        "rho_rep": representative["rho"],
        "tf_num_rep": representative["tf_num"],
        "K_t_f_num_rep": representative["K_t_f_num"],
        "t_obs_rep": representative["t_obs"],
        "K_t_obs_rep": representative["K_t_obs"],
        "R_gap_obs_rep": representative["R_gap_obs"],
        "R_gap_obs_over_rho2_rep": representative["R_gap_obs"] / representative["rho"] ** 2,
        "E_theta_obs_rep": representative["E_theta_obs"],
        "E_phi_obs_rep": representative["E_phi_obs"],
        "E_lock_obs_rep": representative["E_lock_obs"],
        "E_lock_obs_over_rho2_rep": representative["E_lock_obs"] / representative["rho"] ** 2,
        "locked_residual_rep": representative["locked_residual"],
        "sphere_norm_error_rep": representative["sphere_norm_error"],
        "rho_values": np.array([r["rho"] for r in sweep]),
        "R_gap_obs_values": np.array([r["R_gap_obs"] for r in sweep]),
        "E_theta_obs_values": np.array([r["E_theta_obs"] for r in sweep]),
        "E_phi_obs_values": np.array([r["E_phi_obs"] for r in sweep]),
        "E_lock_obs_values": np.array([r["E_lock_obs"] for r in sweep]),
        "R_gap_obs_over_rho2_values": np.array([r["R_gap_obs"] / r["rho"] ** 2 for r in sweep]),
        "E_lock_obs_over_rho2_values": np.array([r["E_lock_obs"] / r["rho"] ** 2 for r in sweep]),
        "tf_num_values": tf_num_values,
        "K_t_f_num_values": np.array([r["K_t_f_num"] for r in sweep]),
        "t_obs_values": t_obs_values,
        "K_t_obs_values": np.array([r["K_t_obs"] for r in sweep]),
        "thresholds": np.array([r["threshold"] for r in sweep]),
        "locked_residual_values": np.array([r["locked_residual"] for r in sweep]),
        "sphere_norm_error_values": np.array([r["sphere_norm_error"] for r in sweep]),
    }
    np.savez_compressed(CACHE_FILE, **data)
    print("Saved:", CACHE_FILE)

#%%
# Diagnostics and summary files
def loglog_fit(x, y):
    log_x = np.log(np.asarray(x, dtype=float))
    log_y = np.log(np.asarray(y, dtype=float))
    coeffs = np.polyfit(log_x, log_y, 1)
    slope_value = float(coeffs[0])
    intercept_value = float(coeffs[1])
    pred = slope_value * log_x + intercept_value
    r2_value = float(1.0 - np.sum((log_y - pred) ** 2) / np.sum((log_y - np.mean(log_y)) ** 2))
    return slope_value, intercept_value, r2_value


rho_values = data["rho_values"]
R_values = data["E_lock_obs_values"]
E_theta_slope, E_theta_intercept, E_theta_fit_r2 = loglog_fit(rho_values, data["E_theta_obs_values"])
E_phi_slope, E_phi_intercept, E_phi_fit_r2 = loglog_fit(rho_values, data["E_phi_obs_values"])
slope, intercept, fit_r2 = loglog_fit(rho_values, R_values)
scaling_rows = [
    {
        "K": K, "rho": rho, "t_f_num": tf, "K_t_f_num": Ktf,
        "t_obs": tobs, "K_t_obs": Ktobs, "t_obs_minus_t_f_num": tobs - tf,
        "K_times_t_obs_minus_t_f_num": K * (tobs - tf),
        "R_gap_obs": Rgap, "R_gap_obs_over_rho2": Rgap / rho**2,
        "E_theta_obs": Etheta, "E_phi_obs": Ephi, "E_lock_obs": Elock,
        "E_lock_obs_over_rho2": Elock / rho**2,
        "threshold": threshold,
        "locked_profile_residual": locked, "sphere_norm_error": sphere,
    }
    for K, rho, tf, Ktf, tobs, Ktobs, Rgap, Etheta, Ephi, Elock, threshold, locked, sphere in zip(
        data["K_values"], rho_values, data["tf_num_values"], data["K_t_f_num_values"],
        data["t_obs_values"], data["K_t_obs_values"], data["R_gap_obs_values"],
        data["E_theta_obs_values"], data["E_phi_obs_values"], data["E_lock_obs_values"],
        data["thresholds"], data["locked_residual_values"], data["sphere_norm_error_values"]
    )
]
with (SUMMARY_DIR / "phase_gap_scaling_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(scaling_rows[0]))
    writer.writeheader()
    writer.writerows(scaling_rows)

max_residual = float(np.max(np.abs(data["residual_obs"])))
rho_rep = float(data["rho_rep"])
min_t_obs_gap = float(np.min(data["t_obs_values"] - data["tf_num_values"]))
min_K_t_obs_gap = float(np.min(data["K_values"] * (data["t_obs_values"] - data["tf_num_values"])))
with (SUMMARY_DIR / "phase_gap_prediction_summary.txt").open("w", encoding="utf-8") as stream:
    stream.write(
        "Phase-gap prediction diagnostic\n"
        f"N = {N}\nK = {K_REP:g}\nsigma_omega = {SIGMA_OMEGA:g}\n"
        f"rho = {rho_rep:.12g}\nt_f_num = {float(data['tf_num_rep']):.12g}\n"
        f"K_t_f_num = {float(data['K_t_f_num_rep']):.12g}\n"
        f"observation_rule = {OBSERVATION_RULE}\n"
        f"observation_factor = {OBSERVATION_FACTOR:.12g}\n"
        f"t_obs = {float(data['t_obs_rep']):.12g}\n"
        f"K_t_obs = {float(data['K_t_obs_rep']):.12g}\n"
        f"t_obs - t_f_num = {float(data['t_obs_rep'] - data['tf_num_rep']):.12g}\n"
        f"max residual = {max_residual:.12g}\n"
        f"max residual / rho^2 = {max_residual / rho_rep**2:.12g}\n"
        f"R_gap_obs / rho^2 range = {min(r['R_gap_obs_over_rho2'] for r in scaling_rows):.12g}, "
        f"{max(r['R_gap_obs_over_rho2'] for r in scaling_rows):.12g}\n"
        f"E_lock_obs / rho^2 range = {min(r['E_lock_obs_over_rho2'] for r in scaling_rows):.12g}, "
        f"{max(r['E_lock_obs_over_rho2'] for r in scaling_rows):.12g}\n"
        f"E_theta_obs log-log fitted slope = {E_theta_slope:.12g}\n"
        f"E_theta_obs log-log fit intercept = {E_theta_intercept:.12g}\n"
        f"E_theta_obs log-log fit R^2 = {E_theta_fit_r2:.12g}\n"
        f"E_phi_obs log-log fitted slope = {E_phi_slope:.12g}\n"
        f"E_phi_obs log-log fit intercept = {E_phi_intercept:.12g}\n"
        f"E_phi_obs log-log fit R^2 = {E_phi_fit_r2:.12g}\n"
        f"E_lock_obs log-log fitted slope = {slope:.12g}\n"
        f"E_lock_obs log-log fit intercept = {intercept:.12g}\n"
        f"E_lock_obs log-log fit R^2 = {fit_r2:.12g}\n"
        f"minimum t_obs - t_f_num = {min_t_obs_gap:.12g}\n"
        f"minimum K*(t_obs - t_f_num) = {min_K_t_obs_gap:.12g}\n"
        f"locked-profile residual = {float(data['locked_residual_rep']):.12g}\n"
        f"max_i |delta_i/K - vartheta_i| = "
        f"{float(np.max(np.abs(data['delta'] / K_REP - data['vartheta']))):.12g}\n"
        f"sphere_norm_error = {float(data['sphere_norm_error_rep']):.12g}\n"
        f"precision_mode = {PRECISION_MODE}\n"
        f"jax_x64 = {bool(jax.config.jax_enable_x64)}\n"
        f"rtol = {RTOL:.12g}\natol = {ATOL:.12g}\n"
        f"time_dtype = {_as_cache_string(data['time_dtype'])}\n"
        f"state_dtype = {_as_cache_string(data['state_dtype'])}\n"
        f"theta0 = {THETA0:g}\nphi0 = {PHI0:g}\nc_init = {C_INIT:g}\nC_tol = {C_TOL:g}\n"
    )
save_metadata(SUMMARY_DIR / "metadata_exp02.json", {
    "figure": "Figure 2", "cache_file": CACHE_FILE.relative_to(PROJECT_ROOT), "N": N, "K_representative": K_REP,
    "K_values": K_VALUES, "sigma_omega": SIGMA_OMEGA, "theta0": THETA0,
    "phi0": PHI0, "c_init": C_INIT, "C_tol": C_TOL,
    "schema_version": SCHEMA_VERSION,
    "precision_mode": PRECISION_MODE,
    "jax_x64": bool(jax.config.jax_enable_x64),
    "rtol": RTOL,
    "atol": ATOL,
    "time_dtype": _as_cache_string(data["time_dtype"]),
    "state_dtype": _as_cache_string(data["state_dtype"]),
    "observation_rule": OBSERVATION_RULE,
    "observation_factor": OBSERVATION_FACTOR,
    "rho": rho_values,
    "t_f_num": data["tf_num_values"],
    "K_t_f_num": data["K_t_f_num_values"],
    "t_obs": data["t_obs_values"],
    "K_t_obs": data["K_t_obs_values"],
    "R_gap_obs": data["R_gap_obs_values"],
    "R_gap_obs_over_rho2": data["R_gap_obs_over_rho2_values"],
    "E_theta_obs": data["E_theta_obs_values"],
    "E_phi_obs": data["E_phi_obs_values"],
    "E_lock_obs": data["E_lock_obs_values"],
    "E_lock_obs_over_rho2": data["E_lock_obs_over_rho2_values"],
    "locked_profile_residual": data["locked_residual_values"],
    "sphere_norm_error": data["sphere_norm_error_values"],
    "E_theta_obs_fitted_slope": E_theta_slope,
    "E_theta_obs_fit_intercept": E_theta_intercept,
    "E_theta_obs_fit_R2": E_theta_fit_r2,
    "E_phi_obs_fitted_slope": E_phi_slope,
    "E_phi_obs_fit_intercept": E_phi_intercept,
    "E_phi_obs_fit_R2": E_phi_fit_r2,
    "E_lock_obs_fitted_slope": slope,
    "E_lock_obs_fit_intercept": intercept,
    "E_lock_obs_fit_R2": fit_r2,
    "all_t_obs_post_fast": bool(np.all(data["t_obs_values"] > data["tf_num_values"])),
    "min_t_obs_minus_t_f_num": min_t_obs_gap,
    "min_K_times_t_obs_minus_t_f_num": min_K_t_obs_gap,
})

#%%
# Plot. All layout constants are in the parameter cell above.
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(
    nrows=2, ncols=2, left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
    wspace=WSPACE, hspace=HSPACE, height_ratios=HEIGHT_RATIOS,
)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

lo = min(float(np.min(data["vartheta"])), float(np.min(data["a_obs"])))
hi = max(float(np.max(data["vartheta"])), float(np.max(data["a_obs"])))
pad = 0.1 * (hi - lo)
lo, hi = lo - pad, hi + pad
ax_a.scatter(data["vartheta"], data["a_obs"], s=24, color="#0072B2", edgecolor="0.15", linewidth=0.25)
ax_a.plot([lo, hi], [lo, hi], "--", color="0.20", lw=1.4)
ax_a.set(xlim=(lo, hi), ylim=(lo, hi), ylabel=r"Observed gap at $t_{\mathrm{obs}}$")
ax_a.set_xlabel(r"Locked profile $\vartheta_i$", labelpad=XLABEL_PAD)
ax_a.set_aspect("equal", adjustable="box")

indices = np.arange(1, len(data["residual_obs"]) + 1)
res_limit = 1.15 * float(np.max(np.abs(data["residual_obs"])))
ax_b.plot(indices, data["residual_obs"], marker="o", color="#0072B2")
ax_b.axhline(0.0, color="0.20", linestyle="--", lw=1.4)
ax_b.set(xlim=(1, len(indices)), ylim=(-res_limit, res_limit), ylabel=r"Residual at $t_{\mathrm{obs}}$")
ax_b.set_xlabel(r"Oscillator index $i$", labelpad=XLABEL_PAD)

order = np.argsort(rho_values)
C_ref = float(np.median(R_values / rho_values**2))
ax_c.loglog(rho_values[order], R_values[order], "o-", color="#0072B2", label="Post-fast locking error")
ax_c.loglog(rho_values[order], C_ref * rho_values[order] ** 2, "--", color="0.20",
            label=r"Reference slope $\rho^2$")
ax_c.set_ylabel(r"$E_{\mathrm{lock}}(t_{\mathrm{obs}})$")
ax_c.set_xlabel(r"Spread parameter $\rho$", labelpad=XLABEL_PAD)
ax_c.legend(loc="best")
ax_c.text(0.04, 0.76, rf"Fit slope $={slope:.2f}$", transform=ax_c.transAxes, fontsize=14)
for ax in (ax_a, ax_b, ax_c):
    format_axes(ax)

# First draw once, then align panel (c) to the top-row plotting block.
fig.canvas.draw()
pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
ax_c.set_position([pos_a.x0, pos_c.y0, pos_b.x1 - pos_a.x0, pos_c.height])

# Add panel labels after the final axes positions are fixed.
def add_bottom_panel_label(ax, label, y):
    ax.text(
        0.5, y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
    )

add_bottom_panel_label(ax_a, r"(a)", PANEL_LABEL_Y_TOP)
add_bottom_panel_label(ax_b, r"(b)", PANEL_LABEL_Y_TOP)
add_bottom_panel_label(ax_c, r"(c)", PANEL_LABEL_Y_BOTTOM)

fig.canvas.draw()

#%%
# Save and show
saved_paths = save_figure_all_formats(fig, FIGURE_DIR, "fig2_phase_gap_prediction")
print("Saved:", *saved_paths, sep="\n")
record_figure_regression(PROJECT_ROOT, figure_script="fig02_phase_gap_prediction.py", generated_artifacts=saved_paths)
plt.close(fig)

# %%
