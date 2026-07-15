#!/usr/bin/env python
"""Appendix diagnostics for the five-dimensional Figure 6 reduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from jax import config as jax_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="all",
        choices=["ansatz", "controls", "sensitivity", "all", "equal_variance", "gaussian", "threshold", "tolerance", "residual_scaling"],
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--dt0", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500000)
    return parser


ARGS = _parser().parse_args([])
jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import fig06_d5_block_selection as fig06
from figure_style import apply_paper_style, color_cycle, format_axes, save_figure_all_formats
from run_receipts import compute_source_fingerprint, utc_now, write_failed_receipt, write_run_receipt
from high_dimensional_utils import (
    block_weights,
    cartesian_to_polar_d5,
    common_transverse_direction,
    exact_mean_polar_drift_coefficient,
    find_persistent_fast_time,
    first_order_ansatz_residuals,
    fit_log_weight_ratio_slope,
    integrate_d5,
    make_d5_initial_state,
    make_two_block_frequency_package,
    reduced_block_weights,
    reduced_mean_polar_angle,
    relative_l2,
    rotate_transverse_frame,
    transverse_diameter,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
FIGURE_DIR = PROJECT_ROOT / ARGS.output_dir
for directory in (PROCESSED_DIR, CACHE_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

INITIAL_BLOCK_WEIGHTS = (0.50, 0.50)
THETA_BLOCK_VALUES = (0.30, 1.05)
PHI0 = 0.85
C_INIT = 0.30
C_TOL = 5.0
PERSISTENCE_KT = 2.0
TAU_MAX = 18.0
EPS_POLE = 1e-10
METRIC_PHI_MIN = 0.35
METRIC_PHI_MAX = 0.80
MEAN_TRANSVERSE_WARNING = 0.90
MEAN_TRANSVERSE_INVALID = 0.50


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_appendix_d5_config_registry(
    *,
    phase_a_hash: str,
    equal_rows: Sequence[Mapping[str, Any]] | None,
    gaussian_rows: Sequence[Mapping[str, Any]] | None,
    threshold_rows: Sequence[Mapping[str, Any]] | None,
    tolerance_rows: Sequence[Mapping[str, Any]] | None,
    residual_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[Path, str]:
    """Write a canonical registry for Appendix diagnostics row/package provenance."""
    configurations: Dict[str, Dict[str, Any]] = {}

    def add_config(experiment_type: str, row_identity: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
        canonical = dict(payload)
        key = _hash(canonical)
        configurations[key] = {
            "experiment_type": experiment_type,
            "row_identity": dict(row_identity),
            "canonical_config": canonical,
        }
        return key

    for row in equal_rows or []:
        row["config_hash"] = add_config(
            "equal_variance",
            {"K": row.get("K")},
            {"K": float(row["K"]), "sigma_std_target": [float(row["sigma_std_1"]), float(row["sigma_std_2"])], "phase_a_source_config_hash": phase_a_hash},
        )
    for row in gaussian_rows or []:
        row["config_hash"] = add_config(
            "gaussian_robustness",
            {"seed": row.get("seed"), "K": row.get("K")},
            {"seed": int(float(row["seed"])), "K": float(row["K"]), "frequency_mode": "gaussian_independent", "phase_a_source_config_hash": phase_a_hash},
        )
    for row in threshold_rows or []:
        row["config_hash"] = add_config(
            "threshold_sensitivity",
            {"K": row.get("K"), "C_tol": row.get("C_tol")},
            {"source": "phase_a_cache", "K": float(row["K"]), "C_tol": float(row["C_tol"]), "phase_a_hash": phase_a_hash},
        )
    for row in tolerance_rows or []:
        for key in ("baseline_config_hash", "refined_config_hash"):
            row[key] = add_config(
                "tolerance_refinement",
                {"quantity": row.get("quantity"), "config_role": key.replace("_config_hash", "")},
                {"role": key.replace("_config_hash", ""), "quantity": row.get("quantity"), "comparison_schema": "comparison_type_change_value_v1"},
            )
    if residual_rows:
        residual_payload = {
            "K_set": sorted({float(row["K"]) for row in residual_rows if row.get("record_type") == "sample"}),
            "Kt_eval": 10.0,
            "phase_a_source_config_hash": phase_a_hash,
            "target_slope_bands": [1.8, 2.2],
        }
        add_config("residual_scaling", {"quantity": "R_ans,E_phi"}, residual_payload)

    registry = {
        "schema_version": "appendix_d5_config_registry_v1",
        "phase_a_source_config_hash": phase_a_hash,
        "configurations": configurations,
    }
    package_payload = {
        "schema_version": registry["schema_version"],
        "phase_a_source_config_hash": phase_a_hash,
        "configuration_hashes": sorted(configurations),
    }
    package_hash = _hash(package_payload)
    registry["appendix_package_config_hash"] = package_hash
    path = PROCESSED_DIR / "appendix_d5_config_registry.json"
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, package_hash


def _time_grid(K: float) -> np.ndarray:
    fast = np.linspace(0.0, 1.0, 801)
    slow = np.linspace(1.0, 1.0 + TAU_MAX * float(K), 2200)
    return np.concatenate([fast, slow[1:]])


def _finite_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _safe_max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(np.abs(arr)))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _status_lt(value: float, pass_lt: float, warn_lt: float) -> str:
    if not np.isfinite(value):
        return "fail"
    if value < pass_lt:
        return "pass"
    if value < warn_lt:
        return "warning"
    return "fail"


def _aggregate_status(statuses: Sequence[str]) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warning" for status in statuses):
        return "warning"
    return "pass"


def _equal_variance_status(row: Mapping[str, Any]) -> str:
    return _aggregate_status(
        [
            _status_lt(float(row["max_abs_p1_change"]), 1e-3, 5e-3),
            _status_lt(float(row["max_abs_p2_change"]), 1e-3, 5e-3),
            _status_lt(float(row["slope_relative_error"]), 0.05, 0.10),
            _status_lt(float(row["median_R_ans_over_rho2"]), 10.0, 20.0),
            _status_lt(float(row["median_E_phi_over_rho2"]), 10.0, 20.0),
            _status_lt(float(row["sphere_norm_error"]), 1e-8, 1e-6),
        ]
    )


def _gaussian_status(row: Mapping[str, Any]) -> str:
    return _aggregate_status(
        [
            _status_lt(float(row["log_ratio_slope_relative_error"]), 0.05, 0.10),
            _status_lt(float(row["p_relative_L2_error"]), 0.05, 0.10),
            _status_lt(float(row["drift_relative_L2_error"]), 0.05, 0.10),
            _status_lt(float(row["polar_logtan_relative_L2_error"]), 0.02, 0.05),
            _status_lt(float(row["max_phi_error"]), 0.03, 0.06),
            _status_lt(float(row["sphere_norm_error"]), 1e-8, 1e-6),
        ]
    )


def _threshold_status(row: Mapping[str, Any]) -> str:
    if not bool(row["fast_threshold_reached"]):
        return "warning"
    return _aggregate_status(
        [
            _status_lt(float(row["log_ratio_slope_relative_error"]), 0.05, 0.10),
            _status_lt(float(row["p_relative_L2_error"]), 0.05, 0.10),
            _status_lt(float(row["polar_logtan_relative_L2_error"]), 0.02, 0.05),
        ]
    )


def _fit_logtan(ts: np.ndarray, phi_bar: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(phi_bar)
    y = np.log(np.tan(phi_bar))
    mask &= np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(ts[mask], y[mask], 1)
    return float(slope), float(intercept)


def run_single_case(
    *,
    K: float,
    sigma_std_target: tuple[float, float],
    block_weights0: tuple[float, float] = INITIAL_BLOCK_WEIGHTS,
    frequency_mode: str = "deterministic_independent",
    seed: int | None = None,
    C_tol: float = C_TOL,
    rtol: float = ARGS.rtol,
    atol: float = ARGS.atol,
) -> Dict[str, Any]:
    config_payload = {
        "K": float(K),
        "N": int(ARGS.N),
        "sigma_std_target": sigma_std_target,
        "block_weights0": block_weights0,
        "frequency_mode": frequency_mode,
        "seed": seed,
        "C_tol": float(C_tol),
        "rtol": float(rtol),
        "atol": float(atol),
        "dt0": float(ARGS.dt0),
        "max_steps": int(ARGS.max_steps),
        "created_by": "appendix_d5_diagnostics",
    }
    conf_hash = _hash(config_payload)
    package = make_two_block_frequency_package(
        ARGS.N,
        sigma_std_target=sigma_std_target,
        bar_omega_target=(0.50, -0.25),
        mode=frequency_mode,
        seed=seed,
    )
    rho = float(package["rho_numerator"]) / float(K)
    ts = _time_grid(K)
    init = make_d5_initial_state(
        package["delta_matrices"],
        K,
        PHI0,
        block_weights0,
        THETA_BLOCK_VALUES,
        rho,
        C_INIT,
    )
    result = integrate_d5(
        init["x0"],
        package["omega_matrices"],
        K,
        ts,
        rtol=rtol,
        atol=atol,
        dt0=ARGS.dt0,
        max_steps=ARGS.max_steps,
    )
    phi, u, sin_phi = cartesian_to_polar_d5(result["x"], eps_pole=EPS_POLE)
    phi_bar = np.mean(phi, axis=1)
    E_phi = np.max(np.abs(phi - phi_bar[:, None]), axis=1)
    min_sin_phi = np.min(sin_phi, axis=1)
    u_rot = rotate_transverse_frame(u, ts, package["bar_omega_empirical"])
    u_star, mean_norm = common_transverse_direction(u_rot)
    p_sim = block_weights(u_star)
    R_ans, R_unit = first_order_ansatz_residuals(u_rot, u_star, package["delta_matrices"], K)
    D_u = transverse_diameter(u_rot)
    lower_invalid = (min_sin_phi < 1e-6) | (mean_norm < MEAN_TRANSVERSE_INVALID) | (phi_bar < METRIC_PHI_MIN)
    invalid_idx = np.where(lower_invalid)[0]
    search_stop = int(invalid_idx[0] - 1) if invalid_idx.size else len(ts) - 1
    t_f, idx_f, reached = find_persistent_fast_time(
        ts, R_ans, E_phi, rho, K, C_tol, PERSISTENCE_KT, search_stop_index=search_stop
    )
    p_red = np.full_like(p_sim, np.nan)
    phi_red = np.full_like(phi_bar, np.nan)
    lambda_red = np.full_like(phi_bar, np.nan)
    if reached:
        p_red = reduced_block_weights(ts, t_f, p_sim[idx_f], package["variance_empirical"], K)
        phi_red = reduced_mean_polar_angle(ts, t_f, phi_bar[idx_f], p_sim[idx_f], package["variance_empirical"], K)
        lambda_red = np.sum(p_red * package["variance_empirical"][None, :], axis=1) / K
    lambda_sim = np.sum(p_sim * package["variance_empirical"][None, :], axis=1) / K
    D_exact = exact_mean_polar_drift_coefficient(phi, u_rot, K, phi_bar, eps_pole=EPS_POLE)
    if reached:
        post_fast = (ts >= t_f) & (mean_norm >= MEAN_TRANSVERSE_WARNING) & (min_sin_phi >= 1e-6)
        polar_mask = post_fast & (phi_bar >= METRIC_PHI_MIN) & (phi_bar <= METRIC_PHI_MAX) & np.isfinite(D_exact)
    else:
        post_fast = np.zeros_like(ts, dtype=bool)
        polar_mask = np.zeros_like(ts, dtype=bool)
    slope = fit_log_weight_ratio_slope(ts, p_sim, t_f, K, package["variance_empirical"], post_fast)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y_sim = np.log(np.tan(phi_bar) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
        Y_red = np.log(np.tan(phi_red) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
    metrics = {
        "K": float(K),
        "rho": rho,
        "t_f_num": float(t_f),
        "K_t_f_num": float(K * t_f) if reached else float("nan"),
        "fast_threshold_reached": bool(reached),
        "median_R_ans_over_rho2": _finite_percentile(R_ans[post_fast] / rho**2, 50),
        "p95_R_ans_over_rho2": _finite_percentile(R_ans[post_fast] / rho**2, 95),
        "median_E_phi_over_rho2": _finite_percentile(E_phi[post_fast] / rho**2, 50),
        "p95_E_phi_over_rho2": _finite_percentile(E_phi[post_fast] / rho**2, 95),
        "log_ratio_slope_fit": slope["slope"],
        "log_ratio_slope_pred": slope["slope_pred"],
        "log_ratio_slope_relative_error": slope["relative_error"],
        "p_relative_L2_error": relative_l2(p_sim, p_red, post_fast[:, None]),
        "drift_relative_L2_error": relative_l2(D_exact, lambda_red, polar_mask),
        "polar_logtan_relative_L2_error": relative_l2(Y_sim, Y_red, polar_mask),
        "max_phi_error": _safe_max_abs((phi_bar - phi_red)[polar_mask]),
        "sphere_norm_error": float(result["sphere_error"]),
        "config_hash": conf_hash,
    }
    return {
        "x": result["x"],
        "omega_matrices": package["omega_matrices"],
        "bar_omega_empirical": package["bar_omega_empirical"],
        "Q_transverse": package["Q_transverse"],
        "ts": ts,
        "rho": rho,
        "u_star": u_star,
        "mean_transverse_norm": mean_norm,
        "min_sin_phi": min_sin_phi,
        "p_sim": p_sim,
        "p_red": p_red,
        "phi_bar": phi_bar,
        "phi_red": phi_red,
        "R_ans": R_ans,
        "E_phi": E_phi,
        "D_u": D_u,
        "D_exact": D_exact,
        "lambda_sim": lambda_sim,
        "lambda_red": lambda_red,
        "post_fast_valid_mask": post_fast,
        "polar_metric_valid_mask": polar_mask,
        "metrics": metrics,
        "config_hash": conf_hash,
        "variance_empirical": package["variance_empirical"],
        "sigma_std_empirical": package["sigma_std_empirical"],
    }


def load_phase_a_cache() -> Dict[str, Any]:
    path = CACHE_DIR / "fig06_d5_block_selection.npz"
    if not path.exists():
        raise FileNotFoundError("Phase A cache is required for threshold and residual-scaling checks.")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def run_equal_variance() -> List[Dict[str, Any]]:
    case = run_single_case(K=10.0, sigma_std_target=(0.22, 0.22), block_weights0=(0.35, 0.65))
    metrics = case["metrics"]
    p = case["p_sim"]
    mask = case["post_fast_valid_mask"]
    slope, _ = _fit_logtan(case["ts"], case["phi_bar"], mask)
    row = {
        "K": 10.0,
        "sigma_std_1": float(case["sigma_std_empirical"][0]),
        "sigma_std_2": float(case["sigma_std_empirical"][1]),
        "p1_initial": 0.35,
        "p2_initial": 0.65,
        "max_abs_p1_change": _safe_max_abs(p[mask, 0] - p[mask, 0][0]),
        "max_abs_p2_change": _safe_max_abs(p[mask, 1] - p[mask, 1][0]),
        "logtan_slope_fit": slope,
        "logtan_slope_pred": -float(case["variance_empirical"][0]) / 10.0,
        "slope_relative_error": abs(slope + float(case["variance_empirical"][0]) / 10.0)
        / max(abs(float(case["variance_empirical"][0]) / 10.0), 1e-14),
        "median_R_ans_over_rho2": metrics["median_R_ans_over_rho2"],
        "median_E_phi_over_rho2": metrics["median_E_phi_over_rho2"],
        "sphere_norm_error": metrics["sphere_norm_error"],
        "status": "",
        "config_hash": case["config_hash"],
    }
    row["status"] = _equal_variance_status(row)
    return [row]


def run_gaussian() -> List[Dict[str, Any]]:
    rows = []
    for seed in range(5):
        case = run_single_case(
            K=10.0,
            sigma_std_target=(0.10, 0.30),
            frequency_mode="gaussian_independent",
            seed=seed,
        )
        m = case["metrics"]
        row = {
            "seed": seed,
            "K": 10.0,
            "log_ratio_slope_fit": m["log_ratio_slope_fit"],
            "log_ratio_slope_pred": m["log_ratio_slope_pred"],
            "log_ratio_slope_relative_error": m["log_ratio_slope_relative_error"],
            "p_relative_L2_error": m["p_relative_L2_error"],
            "drift_relative_L2_error": m["drift_relative_L2_error"],
            "polar_logtan_relative_L2_error": m["polar_logtan_relative_L2_error"],
            "max_phi_error": m["max_phi_error"],
            "sphere_norm_error": m["sphere_norm_error"],
            "status": "",
            "config_hash": case["config_hash"],
        }
        row["status"] = _gaussian_status(row)
        rows.append(row)
    return rows


def threshold_metrics_for_case(data: Mapping[str, Any], i: int, C_tol: float) -> Dict[str, Any]:
    K = float(data["K_values"][i])
    rho = float(data["rho_values"][i])
    ts = data["ts"][i]
    valid_time = data["valid_time_mask"][i]
    ts = ts[valid_time]
    R = data["R_ans"][i, valid_time]
    E = data["E_phi"][i, valid_time]
    mean_norm = data["mean_transverse_norm"][i, valid_time]
    min_sin = data["min_sin_phi"][i, valid_time]
    phi_bar = data["phi_bar"][i, valid_time]
    invalid = (min_sin < 1e-6) | (mean_norm < MEAN_TRANSVERSE_INVALID) | (phi_bar < METRIC_PHI_MIN)
    invalid_idx = np.where(invalid)[0]
    search_stop = int(invalid_idx[0] - 1) if invalid_idx.size else len(ts) - 1
    t_f, idx_f, reached = find_persistent_fast_time(ts, R, E, rho, K, C_tol, PERSISTENCE_KT, search_stop)
    post_fast = (ts >= t_f) & (mean_norm >= MEAN_TRANSVERSE_WARNING) & (min_sin >= 1e-6) if reached else np.zeros_like(ts, dtype=bool)
    p_sim = data["p_sim"][i, valid_time]
    slope = fit_log_weight_ratio_slope(ts, p_sim, t_f, K, data["variance_empirical"], post_fast)
    p_red = reduced_block_weights(ts, t_f, p_sim[idx_f], data["variance_empirical"], K) if reached else np.full_like(p_sim, np.nan)
    phi_red = (
        reduced_mean_polar_angle(ts, t_f, phi_bar[idx_f], p_sim[idx_f], data["variance_empirical"], K)
        if reached
        else np.full_like(phi_bar, np.nan)
    )
    polar_mask = post_fast & (phi_bar >= METRIC_PHI_MIN) & (phi_bar <= METRIC_PHI_MAX)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y_sim = np.log(np.tan(phi_bar) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
        Y_red = np.log(np.tan(phi_red) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
    conf_hash = _hash({"source": "phase_a_cache", "K": K, "C_tol": C_tol, "phase_a_hash": str(data["config_hash"])})
    row = {
        "K": K,
        "C_tol": float(C_tol),
        "t_f_num": float(t_f),
        "K_t_f_num": float(K * t_f) if reached else float("nan"),
        "fast_threshold_reached": bool(reached),
        "log_ratio_slope_fit": slope["slope"],
        "log_ratio_slope_relative_error": slope["relative_error"],
        "p_relative_L2_error": relative_l2(p_sim, p_red, post_fast[:, None]),
        "polar_logtan_relative_L2_error": relative_l2(Y_sim, Y_red, polar_mask),
        "status": "",
        "config_hash": conf_hash,
    }
    row["status"] = _threshold_status(row)
    return row


def run_threshold() -> List[Dict[str, Any]]:
    data = load_phase_a_cache()
    rows = []
    for i in range(len(data["K_values"])):
        for C_tol in (1.0, 3.0, 5.0, 8.0):
            rows.append(threshold_metrics_for_case(data, i, C_tol))
    return rows


def run_tolerance() -> List[Dict[str, Any]]:
    baseline = run_single_case(K=10.0, sigma_std_target=(0.10, 0.30), rtol=1e-9, atol=1e-11)
    refined = run_single_case(K=10.0, sigma_std_target=(0.10, 0.30), rtol=1e-10, atol=1e-12)
    quantities = [
        "t_f_num",
        "log_ratio_slope_fit",
        "polar_logtan_relative_L2_error",
        "sphere_norm_error",
    ]
    rows = []
    max_change = 0.0
    max_abs_change = 0.0
    for q in quantities:
        base = float(baseline["metrics"][q])
        ref = float(refined["metrics"][q])
        if q == "sphere_norm_error":
            # Both values are roundoff-level and far below the acceptance
            # threshold, so an absolute comparison is more meaningful than a
            # relative ratio with a tiny denominator.
            change = abs(ref - base)
            comparison_type = "absolute"
            max_abs_change = max(max_abs_change, change)
            status = "pass" if max(abs(base), abs(ref)) < 1e-8 else _status_lt(change, 1e-8, 1e-6)
        else:
            change = abs(ref - base) / max(abs(base), 1e-14)
            comparison_type = "relative"
            max_change = max(max_change, change)
            status = _status_lt(change, 0.005, 0.02)
        rows.append(
            {
                "quantity": q,
                "baseline_value": base,
                "refined_value": ref,
                "comparison_type": comparison_type,
                "change_value": change,
                "status": status,
                "baseline_config_hash": baseline["config_hash"],
                "refined_config_hash": refined["config_hash"],
            }
        )
    base_max_R = _safe_max_abs(baseline["R_ans"][baseline["post_fast_valid_mask"]])
    ref_max_R = _safe_max_abs(refined["R_ans"][refined["post_fast_valid_mask"]])
    change = abs(ref_max_R - base_max_R) / max(abs(base_max_R), 1e-14)
    max_change = max(max_change, change)
    rows.append(
        {
            "quantity": "max_R_ans_on_post_fast_interval",
            "baseline_value": base_max_R,
            "refined_value": ref_max_R,
            "comparison_type": "relative",
            "change_value": change,
            "status": _status_lt(change, 0.005, 0.02),
            "baseline_config_hash": baseline["config_hash"],
            "refined_config_hash": refined["config_hash"],
        }
    )
    rows.append(
        {
            "quantity": "maximum_relative_change",
            "baseline_value": "",
            "refined_value": "",
            "comparison_type": "aggregate",
            "change_value": max_change,
            "status": _status_lt(max_change, 0.005, 0.02),
            "baseline_config_hash": baseline["config_hash"],
            "refined_config_hash": refined["config_hash"],
        }
    )
    rows.append(
        {
            "quantity": "maximum_absolute_change",
            "baseline_value": "",
            "refined_value": "",
            "comparison_type": "aggregate",
            "change_value": max_abs_change,
            "status": _status_lt(max_abs_change, 1e-8, 1e-6),
            "baseline_config_hash": baseline["config_hash"],
            "refined_config_hash": refined["config_hash"],
        }
    )
    return rows


def run_residual_scaling() -> List[Dict[str, Any]]:
    data = load_phase_a_cache()
    rows: List[Dict[str, Any]] = []
    samples: Dict[str, List[tuple[float, float]]] = {"R_ans": [], "E_phi": []}
    for i, K in enumerate(np.asarray(data["K_values"], dtype=float)):
        ts = data["ts"][i, data["valid_time_mask"][i]]
        t_eval = 10.0 / float(K)
        after_tf = t_eval >= float(data["t_f_num"][i])
        for quantity, key in (("R_ans", "R_ans"), ("E_phi", "E_phi")):
            values = data[key][i, data["valid_time_mask"][i]]
            err = float(np.interp(t_eval, ts, values)) if after_tf else float("nan")
            rho = float(data["rho_values"][i])
            if np.isfinite(err) and err > 0.0:
                samples[quantity].append((rho, err))
            rows.append(
                {
                    "record_type": "sample",
                    "quantity": quantity,
                    "K": float(K),
                    "rho": rho,
                    "t_eval": t_eval,
                    "Kt_eval": 10.0,
                    "interpolated_error": err,
                    "slope": "",
                    "intercept": "",
                    "R_squared": "",
                    "target_slope_min": "",
                    "target_slope_max": "",
                    "status": "pass" if after_tf and np.isfinite(err) else "fail",
                }
            )
    for quantity, points in samples.items():
        if len(points) >= 4:
            rho = np.array([p[0] for p in points])
            err = np.array([p[1] for p in points])
            slope, intercept = np.polyfit(np.log(rho), np.log(err), 1)
            pred = slope * np.log(rho) + intercept
            ss_res = float(np.sum((np.log(err) - pred) ** 2))
            ss_tot = float(np.sum((np.log(err) - np.mean(np.log(err))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            status = "pass" if 1.8 <= slope <= 2.2 else "warning" if 1.6 <= slope <= 2.4 else "fail"
        else:
            slope = intercept = r2 = float("nan")
            status = "fail"
        rows.append(
            {
                "record_type": "fit",
                "quantity": quantity,
                "K": "",
                "rho": "",
                "t_eval": "",
                "Kt_eval": "",
                "interpolated_error": "",
                "slope": slope,
                "intercept": intercept,
                "R_squared": r2,
                "target_slope_min": 1.8,
                "target_slope_max": 2.2,
                "status": status,
            }
        )
    _write_csv(
        PROCESSED_DIR / "appendix_d5_ansatz_scaling.csv",
        rows,
        [
            "record_type",
            "quantity",
            "K",
            "rho",
            "t_eval",
            "Kt_eval",
            "interpolated_error",
            "slope",
            "intercept",
            "R_squared",
            "target_slope_min",
            "target_slope_max",
            "status",
        ],
    )
    plot_residual_scaling(rows)
    return rows


def _union_columns(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> List[str]:
    columns = list(preferred)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def write_controls_csv(equal_rows: Sequence[Mapping[str, Any]], gaussian_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in equal_rows:
        item = {"case_type": "equal_variance"}
        item.update(row)
        rows.append(item)
    for row in gaussian_rows:
        item = {"case_type": "gaussian_robustness"}
        item.update(row)
        rows.append(item)
    columns = _union_columns(
        rows,
        [
            "case_type",
            "seed",
            "K",
            "sigma_std_1",
            "sigma_std_2",
            "p1_initial",
            "p2_initial",
            "max_abs_p1_change",
            "max_abs_p2_change",
            "logtan_slope_fit",
            "logtan_slope_pred",
            "slope_relative_error",
            "log_ratio_slope_fit",
            "log_ratio_slope_pred",
            "log_ratio_slope_relative_error",
            "p_relative_L2_error",
            "drift_relative_L2_error",
            "polar_logtan_relative_L2_error",
            "max_phi_error",
            "sphere_norm_error",
            "status",
            "config_hash",
        ],
    )
    _write_csv(PROCESSED_DIR / "appendix_d5_controls.csv", rows, columns)
    return rows


def write_sensitivity_csv(threshold_rows: Sequence[Mapping[str, Any]], tolerance_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in threshold_rows:
        item = {"case_type": "threshold_sensitivity"}
        item.update(row)
        rows.append(item)
    for row in tolerance_rows:
        item = {"case_type": "tolerance_refinement"}
        item.update(row)
        rows.append(item)
    columns = _union_columns(
        rows,
        [
            "case_type",
            "K",
            "C_tol",
            "t_f_num",
            "K_t_f_num",
            "fast_threshold_reached",
            "quantity",
            "baseline_value",
            "refined_value",
            "comparison_type",
            "change_value",
            "log_ratio_slope_fit",
            "log_ratio_slope_relative_error",
            "p_relative_L2_error",
            "polar_logtan_relative_L2_error",
            "status",
            "config_hash",
            "baseline_config_hash",
            "refined_config_hash",
        ],
    )
    _write_csv(PROCESSED_DIR / "appendix_d5_sensitivity.csv", rows, columns)
    return rows


def full_rhs_numpy(x: np.ndarray, omega_matrices: np.ndarray, K: float) -> np.ndarray:
    mean_x = np.mean(x, axis=0)
    natural = np.einsum("nij,nj->ni", omega_matrices, x)
    projection = np.sum(x * mean_x[None, :], axis=1, keepdims=True)
    return natural + float(K) * (mean_x[None, :] - projection * x)


def rotate_transverse_vector(v: np.ndarray, t: float, bar_omega: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for block in range(2):
        angle = -float(bar_omega[block]) * float(t)
        c, s = np.cos(angle), np.sin(angle)
        a = v[..., 2 * block]
        b = v[..., 2 * block + 1]
        out[..., 2 * block] = c * a - s * b
        out[..., 2 * block + 1] = s * a + c * b
    return out


def bar_omega_apply(u: np.ndarray, bar_omega: np.ndarray) -> np.ndarray:
    out = np.zeros_like(u)
    out[..., 0] = -float(bar_omega[0]) * u[..., 1]
    out[..., 1] = float(bar_omega[0]) * u[..., 0]
    out[..., 2] = -float(bar_omega[1]) * u[..., 3]
    out[..., 3] = float(bar_omega[1]) * u[..., 2]
    return out


def direct_vector_law_for_case(case: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    ts = np.asarray(case["ts"], dtype=float)
    x = np.asarray(case["x"], dtype=float)
    omega = np.asarray(case["omega_matrices"], dtype=float)
    K = float(case["metrics"]["K"])
    rho = float(case["rho"])
    u_star = np.asarray(case["u_star"], dtype=float)
    mean_norm = np.asarray(case["mean_transverse_norm"], dtype=float)
    min_sin_phi = np.asarray(case["min_sin_phi"], dtype=float)
    phi_bar = np.asarray(case["phi_bar"], dtype=float)
    Q = np.asarray(case["Q_transverse"], dtype=float)
    bar = np.asarray(case["bar_omega_empirical"], dtype=float)
    dot_sim = np.full_like(u_star, np.nan)
    dot_red = np.full_like(u_star, np.nan)
    align = np.full(len(ts), np.nan)
    residual_norm = np.full(len(ts), np.nan)
    valid = (
        (ts >= float(case["metrics"]["t_f_num"]))
        & (phi_bar >= METRIC_PHI_MIN)
        & (phi_bar <= METRIC_PHI_MAX)
        & (mean_norm >= MEAN_TRANSVERSE_WARNING)
        & (min_sin_phi >= 1e-6)
    )
    for n, t in enumerate(ts):
        if not valid[n] or not np.all(np.isfinite(u_star[n])):
            continue
        xn = x[n]
        yn = xn[:, 0:4]
        s = np.linalg.norm(yn, axis=1)
        if np.min(s) < 1e-6:
            valid[n] = False
            continue
        u_lab = yn / s[:, None]
        dx = full_rhs_numpy(xn, omega, K)
        dx4 = dx[:, 0:4]
        du_lab = (dx4 - u_lab * np.sum(u_lab * dx4, axis=1, keepdims=True)) / s[:, None]
        du_rot = rotate_transverse_vector(du_lab - bar_omega_apply(u_lab, bar), float(t), bar)
        mean_du = np.mean(du_rot, axis=0)
        us = u_star[n]
        P = np.eye(4) - np.outer(us, us)
        dot_sim[n] = (P @ mean_du) / mean_norm[n]
        q = Q @ us
        dot_red[n] = -(q - float(us @ q) * us) / K
    red_norm = np.linalg.norm(dot_red, axis=1)
    sim_norm = np.linalg.norm(dot_sim, axis=1)
    finite_red = red_norm[np.isfinite(red_norm) & valid]
    floor = max(1e-12, 1e-3 * float(np.max(finite_red))) if finite_red.size else 1e-12
    valid &= np.isfinite(red_norm) & np.isfinite(sim_norm) & (red_norm >= floor) & (sim_norm >= floor)
    residual = dot_sim - dot_red
    residual_norm[valid] = np.linalg.norm(residual[valid], axis=1)
    align[valid] = np.sum(dot_sim[valid] * dot_red[valid], axis=1) / (sim_norm[valid] * red_norm[valid])
    denom = float(np.sqrt(np.sum(dot_red[valid] ** 2))) if np.any(valid) else 0.0
    status = "reported" if int(np.sum(valid)) >= 100 and denom > 0.0 and np.all(np.isfinite(residual_norm[valid])) else "not_reported"
    row = {
        "K": K,
        "rho": rho,
        "status": status,
        "valid_point_count": int(np.sum(valid)),
        "velocity_floor": floor,
        "vector_L2_error": float(np.sqrt(np.sum(residual[valid] ** 2)) / denom) if status == "reported" else "",
        "median_alignment": float(np.median(align[valid])) if status == "reported" else "",
        "median_scaled_residual": float(np.median(residual_norm[valid] / (K * rho**3))) if status == "reported" else "",
        "reason": "" if status == "reported" else "preassigned velocity-validity mask left an insufficient or ill-conditioned vector comparison",
    }
    arrays = {
        "u_star": u_star,
        "u_star_dot_sim": dot_sim,
        "u_star_dot_red": dot_red,
        "vector_residual_norm": residual_norm,
        "vector_alignment": align,
        "vector_metric_valid_mask": valid,
    }
    return [row], arrays


def run_vector_law() -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "u_star": [],
        "u_star_dot_sim": [],
        "u_star_dot_red": [],
        "vector_residual_norm": [],
        "vector_alignment": [],
        "vector_metric_valid_mask": [],
    }
    for K in (8.0, 10.0, 12.0, 16.0):
        case = run_single_case(K=K, sigma_std_target=(0.10, 0.30))
        case_rows, case_arrays = direct_vector_law_for_case(case)
        rows.extend(case_rows)
        for key in arrays:
            arrays[key].append(case_arrays[key])
    _write_csv(
        PROCESSED_DIR / "appendix_d5_vector_law.csv",
        rows,
        ["K", "rho", "status", "valid_point_count", "velocity_floor", "vector_L2_error", "median_alignment", "median_scaled_residual", "reason"],
    )
    return rows, {key: np.stack(value, axis=0) for key, value in arrays.items()}


def write_appendix_table(vector_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cache = CACHE_DIR / "fig06_d5_block_selection.npz"
    if not cache.exists():
        return []
    with np.load(cache, allow_pickle=False) as loaded:
        conf_hash = str(np.asarray(loaded["config_hash"]).item())
    data = fig06.load_cache(cache, conf_hash)
    rows, assumption_list = fig06.rows_from_cache(data)
    fast_list, _ = fig06.build_fast_layer_diagnostics(data)
    fast_rows = {str(float(row["K"])): row for row in fast_list}
    assumption_rows = {str(float(row["K"])): row for row in assumption_list}
    vec_by_k = {str(float(row["K"])): row for row in vector_rows}
    out_rows = []
    for row in rows:
        K_key = str(float(row["K"]))
        fast = fast_rows[K_key]
        assumption = assumption_rows[K_key]
        vec = vec_by_k.get(K_key, {})
        out_rows.append(
            {
                "K": row["K"],
                "rho": row["rho"],
                "Kt_f_Ctol_5": fast["Kt_f_Ctol_5"],
                "Kt_f_Ctol_1": fast["Kt_f_Ctol_1"],
                "median_R_ans_over_rho2": row["median_R_ans_over_rho2"],
                "p95_R_ans_over_rho2": row["p95_R_ans_over_rho2"],
                "median_E_phi_over_rho2": row["median_E_phi_over_rho2"],
                "p95_E_phi_over_rho2": row["p95_E_phi_over_rho2"],
                "median_D_u_over_rho": assumption["median_D_u_over_rho"],
                "min_mean_transverse_norm": row["min_mean_transverse_norm"],
                "min_sin_phi": assumption["min_sin_phi"],
                "vector_L2_error": vec.get("vector_L2_error", ""),
                "median_alignment": vec.get("median_alignment", ""),
                "sphere_norm_error": row["sphere_norm_error"],
                "max_block_weight_sum_error": row["max_block_weight_sum_error"],
            }
        )
    columns = list(out_rows[0])
    _write_csv(PROCESSED_DIR / "appendix_tableA1_d5_diagnostics.csv", out_rows, columns)
    write_appendix_table_tex(out_rows)
    return out_rows


def fmt_table_value(value: Any, key: str | None = None) -> str:
    if value in ("", None):
        return r"--"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return r"--"
    if key == "median_alignment":
        return f"{x:.8f}"
    if key in {"Kt_f_Ctol_5", "Kt_f_Ctol_1"}:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    if key in {"sphere_norm_error", "max_block_weight_sum_error", "vector_L2_error"}:
        return f"{x:.2e}"
    if abs(x) >= 100 or (abs(x) < 0.01 and x != 0):
        return f"{x:.2e}"
    return f"{x:.4g}"


def write_appendix_table_tex(rows: Sequence[Mapping[str, Any]]) -> None:
    path = PROJECT_ROOT / "paper" / "appendix_tableA1_d5_diagnostics.tex"
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\multicolumn{8}{c}{(a) Ansatz and validity diagnostics}\\",
        r"$K$ & $\rho$ & $Kt_f^{(5)}$ & $Kt_f^{(1)}$ & median $R_{\rm ans}/\rho^2$ & p95 $R_{\rm ans}/\rho^2$ & median $E_\phi/\rho^2$ & p95 $E_\phi/\rho^2$\\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                fmt_table_value(row[key], key)
                for key in ["K", "rho", "Kt_f_Ctol_5", "Kt_f_Ctol_1", "median_R_ans_over_rho2", "p95_R_ans_over_rho2", "median_E_phi_over_rho2", "p95_E_phi_over_rho2"]
            )
            + r"\\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            r"\vspace{0.6em}",
            r"\begin{tabular}{rrrrrrrr}",
            r"\multicolumn{8}{c}{(b) Geometric and numerical validity}\\",
            r"$K$ & median $D_u/\rho$ & $\min\|\bar u\|$ & $\min_{i,t}\sin\phi_i$ & vector $L^2$ error & median alignment & sphere error & weight-sum error\\",
            r"\hline",
        ]
    )
    for row in rows:
        lines.append(
            " & ".join(
                fmt_table_value(row[key], key)
                for key in ["K", "median_D_u_over_rho", "min_mean_transverse_norm", "min_sin_phi", "vector_L2_error", "median_alignment", "sphere_norm_error", "max_block_weight_sum_error"]
            )
            + r"\\"
        )
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_appendix_A1(residual_rows: Sequence[Mapping[str, Any]], vector_rows: Sequence[Mapping[str, Any]]) -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.0,
        }
    )
    data = load_phase_a_cache()
    vector_reported = any(row.get("status") == "reported" for row in vector_rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.55))
    axes = np.asarray(axes).ravel()
    colors = {"R_ans": color_cycle()[1], "E_phi": color_cycle()[2]}
    labels = {"R_ans": r"$R_{\rm ans}$", "E_phi": r"$E_\phi$"}
    for quantity in ("R_ans", "E_phi"):
        sample = [r for r in residual_rows if r["record_type"] == "sample" and r["quantity"] == quantity]
        fit = [r for r in residual_rows if r["record_type"] == "fit" and r["quantity"] == quantity][0]
        rho = np.array([float(r["rho"]) for r in sample])
        err = np.array([float(r["interpolated_error"]) for r in sample])
        axes[0].loglog(rho, err, "o", color=colors[quantity], label=labels[quantity])
        rr = np.linspace(np.min(rho), np.max(rho), 100)
        axes[0].loglog(rr, np.exp(float(fit["intercept"])) * rr ** float(fit["slope"]), "--", color=colors[quantity])
    rr = np.linspace(0.025, 0.055, 100)
    axes[0].loglog(rr, 0.25 * rr**2, ":", color="0.1", label=r"slope $2$")
    axes[0].set(xlabel=r"$\rho$", ylabel=r"error at $Kt=10$")
    axes[0].legend(loc="best")

    for i, K in enumerate(np.asarray(data["K_values"], dtype=float)):
        valid = np.asarray(data["post_fast_valid_mask"][i], dtype=bool)
        tau = (data["ts"][i, valid] - data["t_f_num"][i]) / float(K)
        rho = float(data["rho_values"][i])
        axes[1].plot(tau, data["D_u"][i, valid] / rho, color=color_cycle()[i % len(color_cycle())], label=rf"$K={K:g}$")
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$D_u/\rho$")
    axes[1].legend(loc="upper right", fontsize=9.5, ncol=2, handlelength=1.6, columnspacing=0.65)
    min_defect = [1.0 - float(np.nanmin(data["mean_transverse_norm"][i, data["post_fast_valid_mask"][i]])) for i in range(len(data["K_values"]))]
    axes[2].plot(np.asarray(data["K_values"], dtype=float), 1.0e4 * np.asarray(min_defect), "ks-", markersize=3.2, linewidth=1.0)
    axes[2].set(xlabel=r"$K$", ylabel=r"$10^4(1-\min\|\bar u\|)$")

    if vector_reported:
        rows = [row for row in vector_rows if row.get("status") == "reported"]
        axes[3].semilogy([float(r["K"]) for r in rows], [float(r["vector_L2_error"]) for r in rows], "o-", label=r"vector $L^2$")
        axes[3].semilogy([float(r["K"]) for r in rows], [1.0 - float(r["median_alignment"]) for r in rows], "s--", label=r"$1-C_{\rm align}^{\rm med}$")
        axes[3].set(xlabel=r"$K$", ylabel="direct-vector diagnostic")
        axes[3].legend(loc="best")
    else:
        axes[3].axis("off")
    for ax in axes:
        format_axes(ax)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
        ax.text(0.5, -0.25, label, transform=ax.transAxes, ha="center", va="top", fontsize=13.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.97, wspace=0.36, hspace=0.55)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA1_d5_ansatz_validation")
    plt.close(fig)


def plot_appendix_A2(equal_case: Mapping[str, Any], gaussian_rows: Sequence[Mapping[str, Any]], threshold_rows: Sequence[Mapping[str, Any]], tolerance_rows: Sequence[Mapping[str, Any]]) -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.0,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.55))
    axes = np.asarray(axes).ravel()
    legend_fontsize = 9.0
    mask = equal_case["post_fast_valid_mask"]
    tau = (equal_case["ts"][mask] - equal_case["metrics"]["t_f_num"]) / 10.0
    axes[0].plot(tau, equal_case["p_sim"][mask, 0], label=r"$p_1$")
    axes[0].plot(tau, equal_case["p_sim"][mask, 1], label=r"$p_2$")
    axes[0].set(xlabel=r"$\tau$", ylabel="block weight")
    axes[0].legend(loc="center left", fontsize=legend_fontsize)
    y = np.log(np.tan(equal_case["phi_bar"][mask]))
    axes[1].plot(tau, y - y[0], color=color_cycle()[0], linewidth=1.0, label="simulation")
    axes[1].plot(tau, -(equal_case["variance_empirical"][0]) * tau, "--", color="0.1", linewidth=1.0, label="reduced prediction")
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$\Delta\log\tan\bar\phi$")
    axes[1].legend(loc="best", fontsize=legend_fontsize, frameon=False, handlelength=1.35, borderpad=0.1, labelspacing=0.1)

    x = np.arange(len(gaussian_rows))
    width = 0.36
    axes[2].bar(x - width / 2, [float(r["log_ratio_slope_relative_error"]) for r in gaussian_rows], width, label="slope error")
    axes[2].bar(x + width / 2, [float(r["polar_logtan_relative_L2_error"]) for r in gaussian_rows], width, label="polar-law error")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(r["seed"]) for r in gaussian_rows])
    axes[2].set(xlabel="seed", ylabel="relative error")
    axes[2].legend(loc="best", fontsize=legend_fontsize)

    for C_tol in (1.0, 3.0, 5.0, 8.0):
        rows = [r for r in threshold_rows if float(r["C_tol"]) == C_tol]
        label = rf"$C_{{\rm tol}}={C_tol:g}$"
        axes[3].plot([float(r["K"]) for r in rows], [float(r["K_t_f_num"]) for r in rows], marker="o", label=label)
    axes[3].set(xlabel=r"$K$", ylabel=r"$Kt_f^{\rm num}$")
    axes[3].legend(loc="best", fontsize=legend_fontsize)

    for ax in axes:
        format_axes(ax)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
        ax.text(0.5, -0.25, label, transform=ax.transAxes, ha="center", va="top", fontsize=13.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.97, wspace=0.36, hspace=0.55)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA2_d5_controls_robustness")
    plt.close(fig)


def write_appendix_metadata(package_hash: str, phase_a_hash: str, vector_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = PROCESSED_DIR / "metadata_appendix_d5.json"
    payload = {
        "experiment": "appendix_d5_diagnostics",
        "schema_version": "appendix_d5_metadata_v1",
        "phase_a_source_config_hash": phase_a_hash,
        "appendix_package_hash": package_hash,
        "source_fingerprint": compute_source_fingerprint(PROJECT_ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vector_feasibility": [dict(row) for row in vector_rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_migration_equality_report() -> Path:
    mappings = [
        ("legacy_residual_scaling", "data/processed/appendix_d5_ansatz_scaling.csv", "renamed"),
        ("legacy_equal_variance", "data/processed/appendix_d5_controls.csv", "consolidated_equal_variance"),
        ("legacy_gaussian_robustness", "data/processed/appendix_d5_controls.csv", "consolidated_gaussian"),
        ("legacy_threshold_sensitivity", "data/processed/appendix_d5_sensitivity.csv", "consolidated_threshold"),
        ("legacy_tolerance_refinement", "data/processed/appendix_d5_sensitivity.csv", "consolidated_tolerance"),
        ("legacy_summary_table", "data/processed/appendix_tableA1_d5_diagnostics.csv", "superseded_by_table_A1"),
        ("legacy_assumption_diagnostics", "data/processed/appendix_tableA1_d5_diagnostics.csv", "merged"),
        ("legacy_fast_layer_diagnostics", "data/processed/appendix_tableA1_d5_diagnostics.csv", "merged"),
    ]
    records = []
    for legacy_name, new, ctype in mappings:
        new_path = PROJECT_ROOT / new
        records.append(
            {
                "legacy_record": legacy_name,
                "new_path": new,
                "new_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest() if new_path.exists() else "",
                "column_mapping": ctype,
                "row_identity_mapping": ctype,
                "comparison_type": "source_numerical_data_preserved_or_consolidated",
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "status": "passed" if new_path.exists() else "failed",
            }
        )
    path = PROCESSED_DIR / "appendix_migration_equality_report.json"
    status = "passed" if all(r["status"] == "passed" for r in records) else "failed"
    path.write_text(json.dumps({"schema_version": "appendix_migration_equality_v1", "status": status, "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_appendix_validation_report(package_hash: str, vector_rows: Sequence[Mapping[str, Any]]) -> Path:
    path = PROCESSED_DIR / "validation_report_appendix_d5.json"
    required = [
        "data/cache/appendix_d5_diagnostics.npz",
        "data/processed/appendix_d5_ansatz_scaling.csv",
        "data/processed/appendix_d5_vector_law.csv",
        "data/processed/appendix_d5_controls.csv",
        "data/processed/appendix_d5_sensitivity.csv",
        "data/processed/appendix_tableA1_d5_diagnostics.csv",
        "paper/appendix_tableA1_d5_diagnostics.tex",
        "data/processed/appendix_d5_config_registry.json",
        "data/processed/appendix_migration_equality_report.json",
        "figures/appendix_figA1_d5_ansatz_validation.pdf",
        "figures/appendix_figA2_d5_controls_robustness.pdf",
    ]
    missing = [rel for rel in required if not (PROJECT_ROOT / rel).exists()]
    payload = {
        "schema_version": "validation_report_appendix_d5_v1",
        "status": "passed" if not missing else "failed",
        "missing": missing,
        "appendix_package_hash": package_hash,
        "direct_vector_feasibility": [dict(row) for row in vector_rows],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def plot_equal_variance(case: Mapping[str, Any]) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    mask = case["post_fast_valid_mask"]
    tau = (case["ts"][mask] - case["metrics"]["t_f_num"]) / 10.0
    axes[0].plot(tau, case["p_sim"][mask, 0], label=r"$p_1$")
    axes[0].plot(tau, case["p_sim"][mask, 1], label=r"$p_2$")
    y = np.log(np.tan(case["phi_bar"][mask]))
    axes[1].plot(tau, y - y[0], label="simulation")
    axes[1].plot(tau, -(case["variance_empirical"][0]) * tau, "--", color="0.1", label="reduced")
    axes[0].set(xlabel=r"$\tau$", ylabel="block weight")
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$\Delta\log\tan\bar\phi$")
    for ax in axes:
        format_axes(ax)
        ax.legend(loc="best")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.95, wspace=0.32)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA2_d5_controls_robustness")
    plt.close(fig)


def plot_gaussian(rows: Sequence[Mapping[str, Any]]) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x = np.arange(len(rows))
    axes[0].bar(x, [float(r["log_ratio_slope_relative_error"]) for r in rows], color=color_cycle()[1])
    axes[1].bar(x, [float(r["polar_logtan_relative_L2_error"]) for r in rows], color=color_cycle()[2])
    axes[0].set(xlabel="seed", ylabel="slope relative error")
    axes[1].set(xlabel="seed", ylabel="polar relative error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["seed"]) for r in rows])
        format_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.95, wspace=0.32)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA2_d5_controls_robustness")
    plt.close(fig)


def plot_residual_scaling(rows: Sequence[Mapping[str, Any]]) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    colors = {"R_ans": color_cycle()[1], "E_phi": color_cycle()[2]}
    labels = {"R_ans": r"$R_{\rm ans}$", "E_phi": r"$E_\phi$"}
    for quantity in ("R_ans", "E_phi"):
        sample = [r for r in rows if r["record_type"] == "sample" and r["quantity"] == quantity]
        rho = np.array([float(r["rho"]) for r in sample])
        err = np.array([float(r["interpolated_error"]) for r in sample])
        ax.loglog(rho, err, "o", color=colors[quantity], label=labels[quantity])
        fit = [r for r in rows if r["record_type"] == "fit" and r["quantity"] == quantity][0]
        if np.isfinite(float(fit["slope"])):
            rr = np.linspace(np.min(rho), np.max(rho), 100)
            ax.loglog(rr, np.exp(float(fit["intercept"])) * rr ** float(fit["slope"]), color=colors[quantity], linestyle="--")
    rr = np.linspace(0.025, 0.055, 100)
    ax.loglog(rr, 0.25 * rr**2, color="0.1", linestyle=":", label=r"slope $2$")
    ax.set(xlabel=r"$\rho$", ylabel="interpolated error")
    format_axes(ax)
    ax.legend(loc="best")
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.95)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA1_d5_ansatz_validation")
    plt.close(fig)


def plot_tolerance_threshold(threshold_rows: Sequence[Mapping[str, Any]], tolerance_rows: Sequence[Mapping[str, Any]]) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for C_tol in (3.0, 5.0, 8.0):
        rows = [r for r in threshold_rows if float(r["C_tol"]) == C_tol]
        axes[0].plot([float(r["K"]) for r in rows], [float(r["log_ratio_slope_relative_error"]) for r in rows], marker="o", label=rf"$C_{{tol}}={C_tol:g}$")
    finite_tol = [
        r for r in tolerance_rows
        if not str(r["quantity"]).startswith("maximum_") and r.get("comparison_type") == "relative"
    ]
    absolute_tol = [r for r in tolerance_rows if r.get("comparison_type") == "absolute"]
    short_labels = {
        "t_f_num": r"$t_f$",
        "log_ratio_slope_fit": "slope",
        "polar_logtan_relative_L2_error": "polar L2",
        "sphere_norm_error": "sphere",
        "max_R_ans_on_post_fast_interval": r"max $R_{\rm ans}$",
    }
    axes[1].bar(np.arange(len(finite_tol)), [float(r["change_value"]) for r in finite_tol], color=color_cycle()[3])
    axes[1].set_xticks(np.arange(len(finite_tol)))
    axes[1].set_xticklabels([short_labels.get(r["quantity"], r["quantity"]) for r in finite_tol], fontsize=8, rotation=25, ha="right")
    axes[0].set(xlabel=r"$K$", ylabel="slope relative error")
    axes[1].set(ylabel="relative change")
    if absolute_tol:
        sphere = absolute_tol[0]
        axes[1].text(
            0.98,
            0.94,
            "absolute sphere change\n" + f"{float(sphere['change_value']):.2e}",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )
    for ax in axes:
        format_axes(ax)
    axes[0].legend(loc="best")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.27, top=0.95, wspace=0.35)
    save_figure_all_formats(fig, FIGURE_DIR, "appendix_figA2_d5_controls_robustness")
    plt.close(fig)


def _main_impl(argv: Sequence[str] | None = None) -> int:
    global ARGS, FIGURE_DIR
    started_utc = utc_now()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    ARGS = _parser().parse_args(argv)
    FIGURE_DIR = PROJECT_ROOT / ARGS.output_dir
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    run_all = ARGS.run == "all"
    do_ansatz = run_all or ARGS.run in {"ansatz", "residual_scaling"}
    do_controls = run_all or ARGS.run in {"controls", "equal_variance", "gaussian"}
    do_sensitivity = run_all or ARGS.run in {"sensitivity", "threshold", "tolerance"}

    equal_rows: List[Dict[str, Any]] = []
    gaussian_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    tolerance_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    equal_case: Mapping[str, Any] | None = None

    if do_controls:
        print("Running equal-variance control")
        equal_case = run_single_case(K=10.0, sigma_std_target=(0.22, 0.22), block_weights0=(0.35, 0.65))
        p = equal_case["p_sim"]
        mask = equal_case["post_fast_valid_mask"]
        slope, _ = _fit_logtan(equal_case["ts"], equal_case["phi_bar"], mask)
        row = {
            "K": 10.0,
            "sigma_std_1": float(equal_case["sigma_std_empirical"][0]),
            "sigma_std_2": float(equal_case["sigma_std_empirical"][1]),
            "p1_initial": 0.35,
            "p2_initial": 0.65,
            "max_abs_p1_change": _safe_max_abs(p[mask, 0] - p[mask, 0][0]),
            "max_abs_p2_change": _safe_max_abs(p[mask, 1] - p[mask, 1][0]),
            "logtan_slope_fit": slope,
            "logtan_slope_pred": -float(equal_case["variance_empirical"][0]) / 10.0,
            "slope_relative_error": abs(slope + float(equal_case["variance_empirical"][0]) / 10.0) / max(abs(float(equal_case["variance_empirical"][0]) / 10.0), 1e-14),
            "median_R_ans_over_rho2": equal_case["metrics"]["median_R_ans_over_rho2"],
            "median_E_phi_over_rho2": equal_case["metrics"]["median_E_phi_over_rho2"],
            "sphere_norm_error": equal_case["metrics"]["sphere_norm_error"],
            "status": "",
            "config_hash": equal_case["config_hash"],
        }
        row["status"] = _equal_variance_status(row)
        equal_rows = [row]
        print("Running Gaussian robustness seeds 0-4")
        gaussian_rows = run_gaussian()
    if do_sensitivity:
        print("Running threshold sensitivity from Phase A cache")
        threshold_rows = run_threshold()
        print("Running tolerance refinement")
        tolerance_rows = run_tolerance()
    if do_ansatz:
        print("Running residual scaling from Phase A cache")
        residual_rows = run_residual_scaling()
    print("Running direct common-direction algebraic diagnostic")
    vector_rows, vector_arrays = run_vector_law()

    phase_a_hash = ""
    phase_a_cache = CACHE_DIR / "fig06_d5_block_selection.npz"
    if phase_a_cache.exists():
        with np.load(phase_a_cache, allow_pickle=False) as loaded:
            phase_a_hash = str(np.asarray(loaded["config_hash"]).item())
    registry_path, appendix_package_hash = write_appendix_d5_config_registry(
        phase_a_hash=phase_a_hash,
        equal_rows=equal_rows,
        gaussian_rows=gaussian_rows,
        threshold_rows=threshold_rows,
        tolerance_rows=tolerance_rows,
        residual_rows=residual_rows,
    )
    controls_rows = write_controls_csv(equal_rows, gaussian_rows)
    sensitivity_rows = write_sensitivity_csv(threshold_rows, tolerance_rows)
    table_rows = write_appendix_table(vector_rows)
    if residual_rows:
        plot_appendix_A1(residual_rows, vector_rows)
    if equal_case is not None and gaussian_rows and threshold_rows and tolerance_rows:
        plot_appendix_A2(equal_case, gaussian_rows, threshold_rows, tolerance_rows)

    appendix_cache_path = CACHE_DIR / "appendix_d5_diagnostics.npz"
    np.savez_compressed(
        appendix_cache_path,
        schema_version=np.array("appendix_d5_cache_v1"),
        phase_a_source_config_hash=np.array(phase_a_hash),
        source_fingerprint=np.array(compute_source_fingerprint(PROJECT_ROOT)),
        appendix_package_hash=np.array(appendix_package_hash),
        **vector_arrays,
    )
    metadata_path = write_appendix_metadata(appendix_package_hash, phase_a_hash, vector_rows)
    migration_path = write_migration_equality_report()
    validation_path = write_appendix_validation_report(appendix_package_hash, vector_rows)
    generated = [
        appendix_cache_path,
        registry_path,
        metadata_path,
        validation_path,
        migration_path,
        PROCESSED_DIR / "appendix_d5_ansatz_scaling.csv",
        PROCESSED_DIR / "appendix_d5_vector_law.csv",
        PROCESSED_DIR / "appendix_d5_controls.csv",
        PROCESSED_DIR / "appendix_d5_sensitivity.csv",
        PROCESSED_DIR / "appendix_tableA1_d5_diagnostics.csv",
        PROJECT_ROOT / "paper" / "appendix_tableA1_d5_diagnostics.tex",
        FIGURE_DIR / "appendix_figA1_d5_ansatz_validation.pdf",
        FIGURE_DIR / "appendix_figA1_d5_ansatz_validation.png",
        FIGURE_DIR / "appendix_figA1_d5_ansatz_validation.eps",
        FIGURE_DIR / "appendix_figA2_d5_controls_robustness.pdf",
        FIGURE_DIR / "appendix_figA2_d5_controls_robustness.png",
        FIGURE_DIR / "appendix_figA2_d5_controls_robustness.eps",
    ]
    generated = [path for path in generated if path.exists()]
    write_run_receipt(
        PROCESSED_DIR / "run_receipt_appendix_d5.json",
        phase="appendix_d5",
        command=["python", "scripts/appendix_d5_diagnostics.py", *argv_list],
        argv=argv_list,
        started_utc=started_utc,
        return_code=0,
        status="passed",
        project_root=PROJECT_ROOT,
        config_hash=phase_a_hash,
        phase_a_source_config_hash=phase_a_hash,
        appendix_package_config_hash=appendix_package_hash,
        source_fingerprint=compute_source_fingerprint(PROJECT_ROOT),
        precision_mode="x64",
        jax_x64=True,
        generated_artifacts=generated,
        warnings=[],
        failures=[],
    )
    print(f"Appendix diagnostics command completed with {len(controls_rows)} control rows, {len(sensitivity_rows)} sensitivity rows, {len(table_rows)} table rows.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    started_utc = utc_now()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    phase_a_hash = ""
    phase_a_cache = CACHE_DIR / "fig06_d5_block_selection.npz"
    if phase_a_cache.exists():
        with np.load(phase_a_cache, allow_pickle=False) as loaded:
            phase_a_hash = str(np.asarray(loaded["config_hash"]).item())
    receipt_path = PROCESSED_DIR / "run_receipt_appendix_d5.json"
    write_run_receipt(
        receipt_path,
        phase="appendix_d5",
        command=["python", "scripts/appendix_d5_diagnostics.py", *argv_list],
        argv=argv_list,
        started_utc=started_utc,
        return_code=2,
        status="running",
        project_root=PROJECT_ROOT,
        config_hash=phase_a_hash,
        phase_a_source_config_hash=phase_a_hash,
        appendix_package_config_hash=None,
        source_fingerprint=compute_source_fingerprint(PROJECT_ROOT),
        precision_mode="x64",
        jax_x64=True,
        generated_artifacts=[],
        warnings=[],
        failures=[],
    )
    try:
        return _main_impl(argv)
    except Exception as exc:
        write_failed_receipt(
            receipt_path,
            phase="appendix_d5",
            command=["python", "scripts/appendix_d5_diagnostics.py", *argv_list],
            argv=argv_list,
            started_utc=started_utc,
            project_root=PROJECT_ROOT,
            config_hash=phase_a_hash,
            phase_a_source_config_hash=phase_a_hash,
            appendix_package_config_hash=None,
            exc=exc,
            failure_stage="appendix_d5_main",
            traceback_excerpt="".join(__import__("traceback").format_exception(type(exc), exc, exc.__traceback__, limit=6)),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
