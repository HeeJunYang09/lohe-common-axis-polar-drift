#%%
# Figure 02: locked phase-gap prediction and rho^2 residual scaling

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
RECOMPUTE = False
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
RTOL = 1.0e-7
ATOL = 1.0e-9

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
    result = integrate_sphere_model(
        omega, K, x0, t0=0.0, t1=T1, num_save=NUM_SAVE, rtol=RTOL, atol=ATOL
    )
    diag = locking_diagnostics(result.ts, result.xs, stats["omega_bar"], vartheta)
    threshold = C_TOL * stats["rho"] ** 2
    idx = first_threshold_index(diag["E_lock"], threshold)
    if idx is None:
        raise RuntimeError(f"K={K:g}: fast threshold was not reached.")
    residual = diag["a"][idx] - vartheta
    return {
        "a_tf": diag["a"][idx], "vartheta": vartheta, "residual": residual,
        "delta": delta, "rho": stats["rho"], "tf_num": result.ts[idx],
        "R_gap": np.max(np.abs(residual)), "threshold": threshold,
        "locked_residual": locked_residual,
        "sphere_norm_error": result.stats["sphere_norm_error"],
    }

if CACHE_FILE.exists() and not RECOMPUTE:
    data = dict(np.load(CACHE_FILE))
    print("Loaded:", CACHE_FILE)
else:
    representative = simulate(K_REP)
    sweep = [simulate(K) for K in K_VALUES]
    data = {
        "K_values": K_VALUES,
        "a_tf": representative["a_tf"],
        "vartheta": representative["vartheta"],
        "residual": representative["residual"],
        "delta": representative["delta"],
        "rho_rep": representative["rho"],
        "tf_num_rep": representative["tf_num"],
        "locked_residual_rep": representative["locked_residual"],
        "sphere_norm_error_rep": representative["sphere_norm_error"],
        "rho_values": np.array([r["rho"] for r in sweep]),
        "R_gap_values": np.array([r["R_gap"] for r in sweep]),
        "tf_num_values": np.array([r["tf_num"] for r in sweep]),
        "thresholds": np.array([r["threshold"] for r in sweep]),
        "locked_residual_values": np.array([r["locked_residual"] for r in sweep]),
        "sphere_norm_error_values": np.array([r["sphere_norm_error"] for r in sweep]),
    }
    np.savez_compressed(CACHE_FILE, **data)
    print("Saved:", CACHE_FILE)

#%%
# Diagnostics and summary files
rho_values = data["rho_values"]
R_values = data["R_gap_values"]
slope = float(np.polyfit(np.log(rho_values), np.log(R_values), 1)[0])
scaling_rows = [
    {
        "K": K, "rho": rho, "t_f_num": tf, "R_gap": R,
        "R_gap_over_rho2": R / rho**2, "threshold": threshold,
        "locked_profile_residual": locked, "sphere_norm_error": sphere,
    }
    for K, rho, tf, R, threshold, locked, sphere in zip(
        data["K_values"], rho_values, data["tf_num_values"], R_values,
        data["thresholds"], data["locked_residual_values"], data["sphere_norm_error_values"]
    )
]
with (SUMMARY_DIR / "phase_gap_scaling_data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(scaling_rows[0]))
    writer.writeheader()
    writer.writerows(scaling_rows)

max_residual = float(np.max(np.abs(data["residual"])))
rho_rep = float(data["rho_rep"])
with (SUMMARY_DIR / "phase_gap_prediction_summary.txt").open("w", encoding="utf-8") as stream:
    stream.write(
        "Phase-gap prediction diagnostic\n"
        f"N = {N}\nK = {K_REP:g}\nsigma_omega = {SIGMA_OMEGA:g}\n"
        f"rho = {rho_rep:.12g}\nt_f_num = {float(data['tf_num_rep']):.12g}\n"
        f"max residual = {max_residual:.12g}\n"
        f"max residual / rho^2 = {max_residual / rho_rep**2:.12g}\n"
        f"R_gap / rho^2 range = {min(r['R_gap_over_rho2'] for r in scaling_rows):.12g}, "
        f"{max(r['R_gap_over_rho2'] for r in scaling_rows):.12g}\n"
        f"log-log fitted slope = {slope:.12g}\n"
        f"locked-profile residual = {float(data['locked_residual_rep']):.12g}\n"
        f"max_i |delta_i/K - vartheta_i| = "
        f"{float(np.max(np.abs(data['delta'] / K_REP - data['vartheta']))):.12g}\n"
        f"sphere_norm_error = {float(data['sphere_norm_error_rep']):.12g}\n"
        f"theta0 = {THETA0:g}\nphi0 = {PHI0:g}\nc_init = {C_INIT:g}\nC_tol = {C_TOL:g}\n"
    )
save_metadata(SUMMARY_DIR / "metadata_exp02.json", {
    "figure": "Figure 2", "cache_file": CACHE_FILE.relative_to(PROJECT_ROOT), "N": N, "K_representative": K_REP,
    "K_values": K_VALUES, "sigma_omega": SIGMA_OMEGA, "theta0": THETA0,
    "phi0": PHI0, "c_init": C_INIT, "C_tol": C_TOL, "fitted_slope": slope,
})

#%%
# Plot. All layout constants are in the parameter cell above.
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(
    nrows=2, ncols=2, left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP,
    wspace=WSPACE, hspace=HSPACE, height_ratios=HEIGHT_RATIOS,
)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

lo = min(float(np.min(data["vartheta"])), float(np.min(data["a_tf"])))
hi = max(float(np.max(data["vartheta"])), float(np.max(data["a_tf"])))
pad = 0.1 * (hi - lo)
lo, hi = lo - pad, hi + pad
ax_a.scatter(data["vartheta"], data["a_tf"], s=24, color="#0072B2", edgecolor="0.15", linewidth=0.25)
ax_a.plot([lo, hi], [lo, hi], "--", color="0.20", lw=1.4)
ax_a.set(xlim=(lo, hi), ylim=(lo, hi), ylabel=r"Observed gap $\theta_i-\bar{\theta}$")
ax_a.set_xlabel(r"Locked profile $\vartheta_i$", labelpad=XLABEL_PAD)
ax_a.set_aspect("equal", adjustable="box")

indices = np.arange(1, len(data["residual"]) + 1)
res_limit = 1.15 * float(np.max(np.abs(data["residual"])))
ax_b.plot(indices, data["residual"], marker="o", color="#0072B2")
ax_b.axhline(0.0, color="0.20", linestyle="--", lw=1.4)
ax_b.set(xlim=(1, len(indices)), ylim=(-res_limit, res_limit), ylabel=r"Residual")
ax_b.set_xlabel(r"Oscillator index $i$", labelpad=XLABEL_PAD)

order = np.argsort(rho_values)
C_ref = float(np.median(R_values / rho_values**2))
ax_c.loglog(rho_values[order], R_values[order], "o-", color="#0072B2", label="Numerical residual")
ax_c.loglog(rho_values[order], C_ref * rho_values[order] ** 2, "--", color="0.20",
            label=r"Reference slope $\rho^2$")
ax_c.set_ylabel(r"$R_{\mathrm{gap}}$")
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
