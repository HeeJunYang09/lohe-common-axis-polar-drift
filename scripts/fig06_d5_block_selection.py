#!/usr/bin/env python
"""Figure 6: five-dimensional block selection and polar drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import traceback
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from jax import config as jax_config


def _preparse_precision(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only precision flags before JAX-dependent project imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--float32-debug", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="run full d=5 integrations")
    parser.add_argument("--cache-path", default="data/cache/fig06_d5_block_selection.npz")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--frequency-mode",
        default="deterministic_independent",
        choices=[
            "deterministic_independent",
            "deterministic_correlated",
            "gaussian_independent",
            "gaussian_correlated",
        ],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--k-values", nargs="+", type=float, default=[8.0, 10.0, 12.0, 16.0])
    parser.add_argument("--sigma-std-target", nargs=2, type=float, default=[0.10, 0.30])
    parser.add_argument("--bar-omega-values", nargs=2, type=float, default=[0.50, -0.25])
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--phi0", type=float, default=0.85)
    parser.add_argument("--c-init", type=float, default=0.30)
    parser.add_argument("--C-tol", type=float, default=5.0)
    parser.add_argument("--persistence-Kt", type=float, default=2.0)
    parser.add_argument("--tau-max", type=float, default=18.0)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--dt0", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--x64", action="store_true", help="explicitly assert publication x64 mode")
    parser.add_argument(
        "--float32-debug",
        action="store_true",
        help="debug-only short precision mode that writes no publication artifacts",
    )
    return parser


# Importing this module must not inspect pytest or caller arguments.  The
# publication default is x64; ``main(argv)`` reapplies the selected precision
# from its own parsed argument list before doing any numerical work.
jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import scipy
import jax
import jaxlib

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common_utils import save_metadata
from figure_style import apply_paper_style, color_cycle, format_axes, save_figure_all_formats
from run_receipts import compute_source_fingerprint, utc_now, validate_receipt, write_failed_receipt, write_run_receipt
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
    validate_common_axis_structure,
)


SCHEMA_VERSION = "2.0"
PERTURBATION_SCHEMA_VERSION = "d5_generic_trig_v1"
INITIAL_BLOCK_WEIGHTS = [0.50, 0.50]
THETA_BLOCK_VALUES = [0.30, 1.05]
EPS_POLE = 1e-10
METRIC_PHI_MIN = 0.35
METRIC_PHI_MAX = 0.80
MEAN_TRANSVERSE_WARNING = 0.90
MEAN_TRANSVERSE_INVALID = 0.50
SAVE_FAST_POINTS = 801
SAVE_SLOW_POINTS = 2200
REPRESENTATIVE_K = 10.0

SUMMARY_COLUMNS = [
    "K",
    "rho",
    "t_f_num",
    "K_t_f_num",
    "fast_threshold_reached",
    "fast_layer_not_resolved",
    "median_R_ans_over_rho2",
    "p95_R_ans_over_rho2",
    "median_E_phi_over_rho2",
    "p95_E_phi_over_rho2",
    "D_u_over_rho_median",
    "log_ratio_slope_fit",
    "log_ratio_slope_pred",
    "log_ratio_slope_relative_error",
    "log_ratio_r_squared",
    "p_relative_L2_error",
    "p_max_error",
    "drift_relative_L2_error",
    "drift_max_relative_error",
    "polar_logtan_relative_L2_error",
    "max_phi_error",
    "sphere_norm_error",
    "min_mean_transverse_norm",
    "max_block_weight_sum_error",
    "post_fast_metric_point_count",
    "polar_metric_point_count",
    "warning_count",
]

ASSUMPTION_COLUMNS = [
    "K",
    "rho",
    "median_R_ans_over_rho2",
    "p95_R_ans_over_rho2",
    "median_E_phi_over_rho2",
    "p95_E_phi_over_rho2",
    "median_D_u_over_rho",
    "p95_D_u_over_rho",
    "min_mean_transverse_norm",
    "min_sin_phi",
]

FAST_LAYER_DIAGNOSTIC_COLUMNS = [
    "K",
    "rho",
    "initial_R_ans",
    "initial_E_phi",
    "initial_R_ans_over_rho",
    "initial_E_phi_over_rho",
    "initial_R_ans_over_rho2",
    "initial_E_phi_over_rho2",
    "R_ans_at_Kt10",
    "E_phi_at_Kt10",
    "R_ans_at_Kt10_over_rho2",
    "E_phi_at_Kt10_over_rho2",
    "Kt_f_Ctol_1",
    "Kt_f_Ctol_3",
    "Kt_f_Ctol_5",
    "Kt_f_Ctol_8",
    "fast_threshold_reached_Ctol_1",
    "fast_threshold_reached_Ctol_3",
    "fast_threshold_reached_Ctol_5",
    "fast_threshold_reached_Ctol_8",
]

REQUIRED_CACHE_KEYS = [
    "schema_version",
    "perturbation_schema_version",
    "config_hash",
    "K_values",
    "rho_values",
    "T_end_values",
    "ts",
    "valid_time_mask",
    "frequency_block1",
    "frequency_block2",
    "delta_block1",
    "delta_block2",
    "sigma_std_target",
    "sigma_std_empirical",
    "variance_empirical",
    "bar_omega_target",
    "bar_omega_empirical",
    "Q_transverse",
    "sample_correlation",
    "orthogonalized",
    "max_skew_residual",
    "max_axis_residual",
    "max_commutator_residual",
    "Q_block_error",
    "initial_block_weights",
    "initial_states",
    "phi_bar",
    "phi_red",
    "E_phi",
    "R_ans",
    "R_unit",
    "D_u",
    "mean_transverse_norm",
    "min_sin_phi",
    "p_sim",
    "p_red",
    "D_exact",
    "lambda_sim",
    "lambda_red",
    "t_f_num",
    "t_f_index",
    "fast_threshold_reached",
    "fast_layer_not_resolved",
    "sphere_norm_error",
    "post_fast_valid_mask",
    "polar_metric_valid_mask",
    "precision_mode",
]

if len(REQUIRED_CACHE_KEYS) != len(set(REQUIRED_CACHE_KEYS)):
    raise RuntimeError("REQUIRED_CACHE_KEYS must contain unique names.")


PHASE_A_RECOMPUTE_RECEIPT = PROJECT_ROOT / "data" / "processed" / "run_receipt_phase_a_recompute.json"
PHASE_A_CACHE_RENDER_RECEIPT = PROJECT_ROOT / "data" / "processed" / "run_receipt_phase_a_cache_render.json"
APPENDIX_D5_RECEIPT = PROJECT_ROOT / "data" / "processed" / "run_receipt_appendix_d5.json"
REGRESSION_RECEIPT = PROJECT_ROOT / "data" / "processed" / "run_receipt_regression_fig1_4.json"

PHASE_A_MAIN_FILES = [
    "data/cache/fig06_d5_block_selection.npz",
    "data/processed/metadata_exp06_d5_block_selection.json",
    "data/processed/summary_exp06_d5_block_selection.csv",
    "data/processed/d5_block_selection_summary.txt",
    "data/processed/failures_exp06_d5_block_selection.json",
    "data/processed/release_manifest_d5.txt",
    "figures/fig6_d5_block_selection.pdf",
    "figures/fig6_d5_block_selection.png",
]

PHASE_A_PUBLIC_FILES = [
    "data/cache/fig06_d5_block_selection.npz",
    "data/processed/metadata_exp06_d5_block_selection.json",
    "data/processed/summary_exp06_d5_block_selection.csv",
    "data/processed/d5_block_selection_summary.txt",
    "figures/fig6_d5_block_selection.pdf",
]

FAILURE_COLUMNS = [
    "phase",
    "K",
    "frequency_mode",
    "seed",
    "stage",
    "exception_type",
    "message",
    "traceback_excerpt",
    "safe_to_continue",
    "created_utc",
]


def _resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def get_git_metadata(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return Git commit and dirty-state metadata when available."""
    commit = "unavailable"
    dirty = None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        commit = result.stdout.strip() or "unavailable"
    except Exception:
        commit = "unavailable"
    try:
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dirty = bool(status.stdout.strip())
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _success_thresholds() -> Dict[str, Any]:
    return {
        "sphere_norm_error": {"pass_lt": 1e-8, "warning_lt": 1e-6},
        "block_weight_sum_error": {"pass_lt": 1e-12, "warning_lt": 1e-10},
        "min_mean_transverse_norm": {"pass_ge": 0.95, "invalid_lt": 0.50},
        "metric_point_count": {"pass_ge": 100},
        "median_R_ans_over_rho2": {"pass_lt": 10.0, "warning_lt": 20.0},
        "p95_R_ans_over_rho2": {"pass_lt": 20.0, "warning_lt": 40.0},
        "median_E_phi_over_rho2": {"pass_lt": 10.0, "warning_lt": 20.0},
        "p95_E_phi_over_rho2": {"pass_lt": 20.0, "warning_lt": 40.0},
        "log_ratio_slope_relative_error": {"pass_lt": 0.05, "warning_lt": 0.10},
        "p_relative_L2_error": {"pass_lt": 0.05, "warning_lt": 0.10},
        "drift_relative_L2_error": {"pass_lt": 0.05, "warning_lt": 0.10},
        "polar_logtan_relative_L2_error": {"pass_lt": 0.02, "warning_lt": 0.05},
        "max_phi_error": {"pass_lt": 0.03, "warning_lt": 0.06},
    }


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dimension": 5,
        "N": int(args.N),
        "K_values": [float(k) for k in args.k_values],
        "frequency_mode": args.frequency_mode,
        "seed": args.seed,
        "sigma_std_target": [float(v) for v in args.sigma_std_target],
        "bar_omega_target": [float(v) for v in args.bar_omega_values],
        "phi0": float(args.phi0),
        "initial_block_weights": INITIAL_BLOCK_WEIGHTS,
        "theta_block_values": THETA_BLOCK_VALUES,
        "c_init": float(args.c_init),
        "C_tol": float(args.C_tol),
        "persistence_Kt": float(args.persistence_Kt),
        "tau_max": float(args.tau_max),
        "eps_pole": EPS_POLE,
        "metric_phi_min": METRIC_PHI_MIN,
        "metric_phi_max": METRIC_PHI_MAX,
        "mean_transverse_warning": MEAN_TRANSVERSE_WARNING,
        "mean_transverse_invalid": MEAN_TRANSVERSE_INVALID,
        "save_grid_fast_start": 0.0,
        "save_grid_fast_stop": 1.0,
        "save_grid_fast_points": SAVE_FAST_POINTS,
        "save_grid_slow_start": 1.0,
        "save_grid_slow_stop_rule": "T_end=1+tau_max*K",
        "save_grid_slow_points": SAVE_SLOW_POINTS,
        "save_grid_remove_duplicate_t1": True,
        "perturbation_schema_version": PERTURBATION_SCHEMA_VERSION,
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "dt0": float(args.dt0),
        "max_steps": int(args.max_steps),
        "jax_x64": bool(jax.config.jax_enable_x64),
        "precision_mode": "float32_debug" if args.float32_debug else "x64",
        "success_thresholds": _success_thresholds(),
    }


def config_hash(config_dict: Mapping[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_time_grid(K: float, tau_max: float) -> np.ndarray:
    T_end = 1.0 + float(tau_max) * float(K)
    fast = np.linspace(0.0, 1.0, SAVE_FAST_POINTS)
    slow = np.linspace(1.0, T_end, SAVE_SLOW_POINTS)
    return np.concatenate([fast, slow[1:]])


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _safe_max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _status_lt(value: float, pass_lt: float, warn_lt: float) -> str:
    if not np.isfinite(value):
        return "fail"
    if value < pass_lt:
        return "pass"
    if value < warn_lt:
        return "warning"
    return "fail"


def _status_ge(value: float, pass_ge: float, invalid_lt: float | None = None) -> str:
    if not np.isfinite(value):
        return "fail"
    if value >= pass_ge:
        return "pass"
    if invalid_lt is not None and value < invalid_lt:
        return "fail"
    return "warning"


def _row_warning_count(row: Mapping[str, Any]) -> int:
    checks = [
        _status_lt(float(row["sphere_norm_error"]), 1e-8, 1e-6),
        _status_lt(float(row["max_block_weight_sum_error"]), 1e-12, 1e-10),
        _status_ge(float(row["min_mean_transverse_norm"]), 0.95, 0.50),
        _status_ge(float(row["post_fast_metric_point_count"]), 100.0),
        _status_ge(float(row["polar_metric_point_count"]), 100.0),
        _status_lt(float(row["median_R_ans_over_rho2"]), 10.0, 20.0),
        _status_lt(float(row["p95_R_ans_over_rho2"]), 20.0, 40.0),
        _status_lt(float(row["median_E_phi_over_rho2"]), 10.0, 20.0),
        _status_lt(float(row["p95_E_phi_over_rho2"]), 20.0, 40.0),
        _status_lt(float(row["log_ratio_slope_relative_error"]), 0.05, 0.10),
        _status_lt(float(row["p_relative_L2_error"]), 0.05, 0.10),
        _status_lt(float(row["drift_relative_L2_error"]), 0.05, 0.10),
        _status_lt(float(row["polar_logtan_relative_L2_error"]), 0.02, 0.05),
        _status_lt(float(row["max_phi_error"]), 0.03, 0.06),
    ]
    if not bool(row["fast_threshold_reached"]):
        checks.append("warning")
    return sum(1 for item in checks if item == "warning")


APPENDIX_D5_REQUIRED_FILES = [
    "data/cache/appendix_d5_diagnostics.npz",
    "data/processed/appendix_d5_ansatz_scaling.csv",
    "data/processed/appendix_d5_vector_law.csv",
    "data/processed/appendix_d5_controls.csv",
    "data/processed/appendix_d5_sensitivity.csv",
    "data/processed/appendix_tableA1_d5_diagnostics.csv",
    "paper/appendix_tableA1_d5_diagnostics.tex",
    "data/processed/appendix_d5_config_registry.json",
    "data/processed/metadata_appendix_d5.json",
    "data/processed/validation_report_appendix_d5.json",
    "data/processed/appendix_migration_equality_report.json",
    "figures/appendix_figA1_d5_ansatz_validation.pdf",
    "figures/appendix_figA1_d5_ansatz_validation.png",
    "figures/appendix_figA1_d5_ansatz_validation.eps",
    "figures/appendix_figA2_d5_controls_robustness.pdf",
    "figures/appendix_figA2_d5_controls_robustness.png",
    "figures/appendix_figA2_d5_controls_robustness.eps",
]

APPENDIX_D5_PUBLIC_FILES = [
    "data/cache/appendix_d5_diagnostics.npz",
    "data/processed/appendix_d5_ansatz_scaling.csv",
    "data/processed/appendix_d5_vector_law.csv",
    "data/processed/appendix_d5_controls.csv",
    "data/processed/appendix_d5_sensitivity.csv",
    "data/processed/appendix_tableA1_d5_diagnostics.csv",
    "paper/appendix_tableA1_d5_diagnostics.tex",
    "data/processed/appendix_d5_config_registry.json",
    "data/processed/metadata_appendix_d5.json",
    "figures/appendix_figA1_d5_ansatz_validation.pdf",
    "figures/appendix_figA2_d5_controls_robustness.pdf",
]

APPENDIX_D5_CSV_SCHEMAS = {
    "data/processed/appendix_d5_controls.csv": [
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
        "median_R_ans_over_rho2",
        "median_E_phi_over_rho2",
    ],
    "data/processed/appendix_d5_sensitivity.csv": [
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
    "data/processed/appendix_d5_ansatz_scaling.csv": [
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
    "data/processed/appendix_d5_vector_law.csv": [
        "K",
        "rho",
        "status",
        "valid_point_count",
        "velocity_floor",
        "vector_L2_error",
        "median_alignment",
        "median_scaled_residual",
        "reason",
    ],
    "data/processed/appendix_tableA1_d5_diagnostics.csv": [
        "K",
        "rho",
        "Kt_f_Ctol_5",
        "Kt_f_Ctol_1",
        "median_R_ans_over_rho2",
        "p95_R_ans_over_rho2",
        "median_E_phi_over_rho2",
        "p95_E_phi_over_rho2",
        "median_D_u_over_rho",
        "min_mean_transverse_norm",
        "min_sin_phi",
        "vector_L2_error",
        "median_alignment",
        "sphere_norm_error",
        "max_block_weight_sum_error",
    ],
}


def _csv_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _is_finite_text(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_failure_report(path: Path) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not path.exists():
        return [f"missing {path}"], warnings
    try:
        payload = _read_json(path)
    except Exception as exc:
        return [f"invalid failure JSON {path}: {exc}"], warnings
    if not isinstance(payload, list):
        return [f"failure JSON {path} is not a list"], warnings
    for i, record in enumerate(payload):
        if not isinstance(record, dict):
            errors.append(f"failure record {i} is not an object")
            continue
        missing = [key for key in FAILURE_COLUMNS if key not in record]
        if missing:
            errors.append(f"failure record {i} missing {missing}")
        if not bool(record.get("safe_to_continue", False)):
            errors.append(f"unsafe failure record {i}")
    return errors, warnings


def _band_assessment(K: str, metric: str, value: float, pass_rule: str, warning_rule: str, status: str, message: str) -> Dict[str, Any]:
    return {
        "K": K,
        "metric": metric,
        "observed_value": value,
        "pass_band": pass_rule,
        "warning_band": warning_rule,
        "fail_band": f"outside pass/warning bands",
        "status": status,
        "message": message,
    }


def _assess_lt(K: str, metric: str, value: float, pass_lt: float, warn_lt: float) -> Dict[str, Any]:
    if not np.isfinite(value):
        status, msg = "failed", "nonfinite"
    elif value < pass_lt:
        status, msg = "passed", "within pass band"
    elif value < warn_lt:
        status, msg = "warning", "within warning band"
    else:
        status, msg = "failed", "outside scientific fail threshold"
    return _band_assessment(K, metric, value, f"<{pass_lt:g}", f"[{pass_lt:g},{warn_lt:g})", status, msg)


def _assess_ge_count(K: str, metric: str, value: float, pass_ge: float) -> Dict[str, Any]:
    if not np.isfinite(value):
        status, msg = "failed", "nonfinite"
    elif value >= pass_ge:
        status, msg = "passed", "within pass band"
    elif value > 0:
        status, msg = "warning", "positive but below pass band"
    else:
        status, msg = "failed", "no valid metric points"
    return _band_assessment(K, metric, value, f">={pass_ge:g}", f"(0,{pass_ge:g})", status, msg)


def _assess_ge_norm(K: str, metric: str, value: float) -> Dict[str, Any]:
    if not np.isfinite(value):
        status, msg = "failed", "nonfinite"
    elif value >= 0.95:
        status, msg = "passed", "within pass band"
    elif value >= 0.50:
        status, msg = "warning", "within warning band"
    else:
        status, msg = "failed", "invalid reduced diagnostic regime"
    return _band_assessment(K, metric, value, ">=0.95", "[0.50,0.95)", status, msg)


def assess_phase_a_scientific_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    assessments: List[Dict[str, Any]] = []
    for row in rows:
        K = str(row.get("K"))
        checks = [
            ("median_R_ans_over_rho2", 10.0, 20.0),
            ("p95_R_ans_over_rho2", 20.0, 40.0),
            ("median_E_phi_over_rho2", 10.0, 20.0),
            ("p95_E_phi_over_rho2", 20.0, 40.0),
            ("log_ratio_slope_relative_error", 0.05, 0.10),
            ("p_relative_L2_error", 0.05, 0.10),
            ("drift_relative_L2_error", 0.05, 0.10),
            ("polar_logtan_relative_L2_error", 0.02, 0.05),
            ("max_phi_error", 0.03, 0.06),
            ("sphere_norm_error", 1e-8, 1e-6),
            ("max_block_weight_sum_error", 1e-12, 1e-10),
        ]
        for metric, pass_lt, warn_lt in checks:
            assessments.append(_assess_lt(K, metric, float(row[metric]), pass_lt, warn_lt))
        assessments.append(_assess_ge_count(K, "post_fast_metric_point_count", float(row["post_fast_metric_point_count"]), 100.0))
        assessments.append(_assess_ge_count(K, "polar_metric_point_count", float(row["polar_metric_point_count"]), 100.0))
        assessments.append(_assess_ge_norm(K, "min_mean_transverse_norm", float(row["min_mean_transverse_norm"])))
        reached = str(row.get("fast_threshold_reached")).lower() in {"true", "1"}
        assessments.append(
            _band_assessment(
                K,
                "fast_threshold_reached",
                1.0 if reached else 0.0,
                "reached",
                "unresolved but diagnostics interpretable",
                "passed" if reached else "warning",
                "persistent threshold reached" if reached else "threshold unresolved",
            )
        )
    return assessments


def _file_exists_nonempty(project_root: Path, rel: str) -> bool:
    path = project_root / rel
    return path.exists() and path.stat().st_size > 0


def validate_phase_a_completion(
    project_root: Path = PROJECT_ROOT,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_config_hash: str | None = None,
    expected_source_fingerprint: str | None = None,
    require_local_records: bool = True,
) -> Dict[str, Any]:
    """Content-aware validation for the Phase A Figure 6 publication artifacts."""
    errors: List[str] = []
    warnings: List[str] = []
    checked: List[str] = []
    config_hash_found: str | None = None
    git_commit_found = "unavailable"
    scientific_assessments: List[Dict[str, Any]] = []
    if expected_config_hash is None:
        if expected_config is None:
            expected_args = _build_parser().parse_args([])
            expected_config = build_config(expected_args)
        expected_config_hash = config_hash(expected_config)
    if expected_source_fingerprint is None:
        try:
            expected_source_fingerprint = compute_source_fingerprint(project_root)
        except Exception as exc:
            errors.append(f"could not compute source fingerprint: {exc}")

    required_files = PHASE_A_MAIN_FILES if require_local_records else PHASE_A_PUBLIC_FILES
    for rel in required_files:
        checked.append(rel)
        if not _file_exists_nonempty(project_root, rel):
            errors.append(f"missing or empty {rel}")

    cache_path = project_root / "data/cache/fig06_d5_block_selection.npz"
    data: Dict[str, Any] | None = None
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as loaded:
                data = {key: loaded[key] for key in loaded.files}
            config_hash_found = str(np.asarray(data["config_hash"]).item())
            if expected_config_hash is not None and config_hash_found != expected_config_hash:
                errors.append("cache config_hash does not match expected configuration")
            validate_cache(data, expected_config_hash or config_hash_found)
            if list(np.asarray(data["K_values"], dtype=float)) != [8.0, 10.0, 12.0, 16.0]:
                errors.append("cache K_values are not exactly [8,10,12,16]")
            for i, K in enumerate(np.asarray(data["K_values"], dtype=float)):
                valid = np.asarray(data["valid_time_mask"][i], dtype=bool)
                ts = np.asarray(data["ts"][i, valid], dtype=float)
                if ts.size < 2 or np.any(np.diff(ts) <= 0):
                    errors.append(f"invalid time grid for K={K:g}")
            for key in ("max_skew_residual", "max_axis_residual", "max_commutator_residual", "Q_block_error"):
                value = float(np.asarray(data[key]).item())
                if not np.isfinite(value) or abs(value) > 1e-12:
                    errors.append(f"structural residual {key} invalid: {value}")
            init_norm_error = float(np.nanmax(np.abs(np.linalg.norm(np.asarray(data["initial_states"]), axis=2) - 1.0)))
            if init_norm_error > 1e-12:
                errors.append(f"initial norm error invalid: {init_norm_error}")
            for key, target in (
                ("sigma_std_empirical", np.asarray([0.10, 0.30])),
                ("variance_empirical", np.asarray([0.01, 0.09])),
                ("bar_omega_empirical", np.asarray([0.50, -0.25])),
            ):
                arr = np.asarray(data[key], dtype=float)
                if np.max(np.abs(arr - target)) > 1e-12:
                    errors.append(f"{key} differs from canonical target beyond 1e-12")
            for key in ("rho_values", "t_f_num", "sphere_norm_error"):
                arr = np.asarray(data[key], dtype=float)
                if not np.all(np.isfinite(arr)):
                    errors.append(f"nonfinite cache values in {key}")
        except Exception as exc:
            errors.append(f"cache validation failed: {exc}")
    else:
        errors.append("main cache missing")

    summary_path = project_root / "data/processed/summary_exp06_d5_block_selection.csv"
    if summary_path.exists():
        try:
            found, rows = _csv_rows(summary_path)
            if found != SUMMARY_COLUMNS:
                errors.append("main summary CSV schema mismatch")
            if len(rows) != 4 or sorted(float(row["K"]) for row in rows) != [8.0, 10.0, 12.0, 16.0]:
                errors.append("main summary CSV does not contain exactly four required K rows")
            finite_cols = [
                "rho",
                "t_f_num",
                "log_ratio_slope_relative_error",
                "p_relative_L2_error",
                "drift_relative_L2_error",
                "polar_logtan_relative_L2_error",
                "sphere_norm_error",
            ]
            for row in rows:
                for col in finite_cols:
                    if not _is_finite_text(row.get(col)):
                        errors.append(f"nonfinite {col} in main summary")
            scientific_assessments = assess_phase_a_scientific_rows(rows)
        except Exception as exc:
            errors.append(f"main summary validation failed: {exc}")

    metadata_path = project_root / "data/processed/metadata_exp06_d5_block_selection.json"
    if metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
            required = [
                "experiment",
                "schema_version",
                "dimension",
                "N",
                "K_values",
                "git_commit",
                "git_dirty",
                "config_hash",
                "python_version",
                "numpy_version",
                "scipy_version",
                "jax_version",
                "jaxlib_version",
                "diffrax_version",
                "matplotlib_version",
            ]
            missing = [key for key in required if key not in metadata]
            if missing:
                errors.append(f"metadata missing {missing}")
            git_commit_found = str(metadata.get("git_commit", "unavailable"))
            if config_hash_found is not None and metadata.get("config_hash") != config_hash_found:
                errors.append("metadata config_hash does not match cache")
        except Exception as exc:
            errors.append(f"metadata validation failed: {exc}")

    if require_local_records:
        failure_errors, failure_warnings = _validate_failure_report(
            project_root / "data/processed/failures_exp06_d5_block_selection.json"
        )
        errors.extend(failure_errors)
        warnings.extend(failure_warnings)

        receipt_errors, receipt_warnings = validate_receipt(
            project_root / "data/processed/run_receipt_phase_a_recompute.json",
            phase="phase_a_recompute",
            project_root=project_root,
            expected_config_hash=config_hash_found if expected_config_hash is None else expected_config_hash,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_precision_mode="x64",
            expected_jax_x64=True,
            required_argv_tokens=("--recompute", "--x64"),
            required_artifacts=("data/cache/fig06_d5_block_selection.npz",),
        )
        errors.extend(receipt_errors)
        warnings.extend(receipt_warnings)
        cache_receipt_errors, cache_receipt_warnings = validate_receipt(
            project_root / "data/processed/run_receipt_phase_a_cache_render.json",
            phase="phase_a_cache_render",
            project_root=project_root,
            expected_config_hash=config_hash_found if expected_config_hash is None else expected_config_hash,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_precision_mode="x64",
            expected_jax_x64=True,
            required_artifacts=(
                "data/processed/metadata_exp06_d5_block_selection.json",
                "data/processed/summary_exp06_d5_block_selection.csv",
                "data/processed/failures_exp06_d5_block_selection.json",
                "figures/fig6_d5_block_selection.pdf",
                "figures/fig6_d5_block_selection.png",
                "figures/fig6_d5_block_selection.eps",
            ),
        )
        errors.extend(cache_receipt_errors)
        warnings.extend(cache_receipt_warnings)

    sci_statuses = {item["status"] for item in scientific_assessments}
    sci_failures = [item for item in scientific_assessments if item["status"] == "failed"]
    sci_warnings = [item for item in scientific_assessments if item["status"] == "warning"]
    if sci_failures:
        warnings.extend([f"scientific fail {item['K']} {item['metric']}: {item['observed_value']}" for item in sci_failures])
    elif sci_warnings:
        warnings.extend([f"scientific warning {item['K']} {item['metric']}: {item['observed_value']}" for item in sci_warnings])
    pipeline_status = "failed" if errors else "passed"
    scientific_status = "failed" if sci_failures else "completed with warnings" if sci_warnings or warnings else "passed"
    status = "failed" if errors or sci_failures else scientific_status
    return {
        "status": status,
        "pipeline_status": pipeline_status,
        "scientific_status": scientific_status,
        "overall_status": status,
        "errors": errors,
        "warnings": warnings,
        "scientific_assessments": scientific_assessments,
        "checked_artifacts": checked,
        "config_hash": config_hash_found,
        "git_commit": git_commit_found,
        "source_fingerprint": expected_source_fingerprint,
    }


def validate_appendix_d5_completion(
    project_root: Path = PROJECT_ROOT,
    *,
    require_local_records: bool = True,
) -> Dict[str, Any]:
    """Content-aware Appendix diagnostics completion validation."""
    errors: List[str] = []
    warnings: List[str] = []
    checked: List[str] = []
    required_files = APPENDIX_D5_REQUIRED_FILES if require_local_records else APPENDIX_D5_PUBLIC_FILES
    for rel in required_files:
        checked.append(rel)
        path = project_root / rel
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty {rel}")
    for rel, columns in APPENDIX_D5_CSV_SCHEMAS.items():
        path = project_root / rel
        if not path.exists():
            continue
        found, rows = _csv_rows(path)
        if found != columns:
            errors.append(f"invalid schema {rel}")
            continue
        if not rows:
            errors.append(f"empty CSV {rel}")
            continue
        for row in rows:
            if row.get("status") == "not run":
                errors.append(f"not-run row in {rel}")
            if row.get("status") == "fail":
                errors.append(f"failed row in {rel}")
            if row.get("status") == "warning":
                warnings.append(f"warning row in {rel}")
            if (
                "config_hash" in row
                and not row.get("config_hash")
                and not (row.get("baseline_config_hash") and row.get("refined_config_hash"))
            ):
                errors.append(f"missing config_hash in {rel}")
            for key, value in row.items():
                if key in {
                    "status",
                    "config_hash",
                    "baseline_config_hash",
                    "refined_config_hash",
                    "record_type",
                    "quantity",
                    "comparison_type",
                    "case_type",
                    "reason",
                }:
                    continue
                if value == "":
                    continue
                if key in {"fast_threshold_reached"}:
                    if str(value) not in {"True", "False", "true", "false", "0", "1"}:
                        errors.append(f"invalid boolean field {key} in {rel}")
                    continue
                if not _is_finite_text(value):
                    errors.append(f"nonfinite numeric field {key} in {rel}")
    controls_path = project_root / "data/processed/appendix_d5_controls.csv"
    if controls_path.exists():
        _, rows = _csv_rows(controls_path)
        equal_rows = [row for row in rows if row.get("case_type") == "equal_variance"]
        gaussian_rows = [row for row in rows if row.get("case_type") == "gaussian_robustness"]
        if len(equal_rows) != 1:
            errors.append("equal-variance control row count is not one")
        if sorted(int(float(row["seed"])) for row in gaussian_rows) != [0, 1, 2, 3, 4]:
            errors.append("Gaussian robustness seeds are not exactly 0,1,2,3,4")
    sensitivity_path = project_root / "data/processed/appendix_d5_sensitivity.csv"
    if sensitivity_path.exists():
        _, rows = _csv_rows(sensitivity_path)
        threshold_rows = [row for row in rows if row.get("case_type") == "threshold_sensitivity"]
        tolerance_rows = [row for row in rows if row.get("case_type") == "tolerance_refinement"]
        pairs = sorted((float(row["K"]), float(row["C_tol"])) for row in threshold_rows)
        expected_pairs = sorted((K, C) for K in [8.0, 10.0, 12.0, 16.0] for C in [1.0, 3.0, 5.0, 8.0])
        if pairs != expected_pairs:
            errors.append("threshold sensitivity rows are not exactly K=[8,10,12,16] x C_tol=[1,3,5,8]")
        quantities = {row.get("quantity") for row in tolerance_rows}
        if "maximum_relative_change" not in quantities:
            errors.append("tolerance maximum_relative_change row missing")
        if "maximum_absolute_change" not in quantities:
            errors.append("tolerance maximum_absolute_change row missing")
        for row in tolerance_rows:
            if row.get("comparison_type") not in {"relative", "absolute", "aggregate"}:
                errors.append("invalid tolerance comparison_type")
            if row.get("baseline_config_hash") and row.get("baseline_config_hash") == row.get("refined_config_hash"):
                errors.append("tolerance baseline/refined config hash collision")
    vector_path = project_root / "data/processed/appendix_d5_vector_law.csv"
    if vector_path.exists():
        _, rows = _csv_rows(vector_path)
        if sorted(float(row["K"]) for row in rows) != [8.0, 10.0, 12.0, 16.0]:
            errors.append("vector-law diagnostic does not contain K=[8,10,12,16]")
        for row in rows:
            if row.get("status") == "reported":
                if int(float(row.get("valid_point_count", "0"))) < 100:
                    errors.append("reported vector-law row has fewer than 100 valid points")
                for key in ("vector_L2_error", "median_alignment", "median_scaled_residual"):
                    if not _is_finite_text(row.get(key, "")):
                        errors.append(f"nonfinite {key} in vector-law row")
            elif not row.get("reason"):
                errors.append("not_reported vector-law row lacks a reason")
    table_path = project_root / "data/processed/appendix_tableA1_d5_diagnostics.csv"
    if table_path.exists():
        _, rows = _csv_rows(table_path)
        if len(rows) != 4:
            errors.append("Appendix Table A.1 CSV row count is not four")
    table_tex = project_root / "paper/appendix_tableA1_d5_diagnostics.tex"
    if table_tex.exists():
        text = table_tex.read_text(encoding="utf-8")
        if "(a)" not in text or "(b)" not in text:
            errors.append("Appendix Table A.1 TeX does not contain both panel labels")
        if "\\begin{table}" in text or "\\end{table}" in text:
            errors.append("Appendix Table A.1 TeX should not wrap itself in a table float")
    phase_a_hash = None
    cache_path = project_root / "data/cache/fig06_d5_block_selection.npz"
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as loaded:
                phase_a_hash = str(np.asarray(loaded["config_hash"]).item())
        except Exception as exc:
            errors.append(f"could not read Phase A cache hash: {exc}")
    registry_path = project_root / "data/processed/appendix_d5_config_registry.json"
    appendix_d5_package_hash = None
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if registry.get("schema_version") != "appendix_d5_config_registry_v1":
                errors.append("Appendix diagnostics registry schema mismatch")
            if phase_a_hash is not None and registry.get("phase_a_source_config_hash") != phase_a_hash:
                errors.append("Appendix diagnostics registry Phase A source hash mismatch")
            configs = registry.get("configurations", {})
            for stored_hash, payload in configs.items():
                canonical = payload.get("canonical_config", {})
                recomputed = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                # Older exact run_single_case hashes are accepted through registry presence;
                # registry-native payloads must still be self-consistent when marked.
                if payload.get("registry_native_hash") and recomputed != stored_hash:
                    errors.append(f"Appendix diagnostics registry hash mismatch for {stored_hash}")
            appendix_d5_package_hash = registry.get("appendix_package_config_hash")
            package_payload = {
                "schema_version": registry.get("schema_version"),
                "phase_a_source_config_hash": registry.get("phase_a_source_config_hash"),
                "configuration_hashes": sorted(configs),
            }
            recomputed_package = hashlib.sha256(json.dumps(package_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if appendix_d5_package_hash != recomputed_package:
                errors.append("Appendix diagnostics package hash mismatch")
            # Every row hash that exists in CSV outputs must be registered.
            registered = set(configs)
            for rel in [
                "data/processed/appendix_d5_controls.csv",
                "data/processed/appendix_d5_sensitivity.csv",
            ]:
                path = project_root / rel
                if path.exists():
                    _, rows = _csv_rows(path)
                    for row in rows:
                        if row.get("config_hash") and row.get("config_hash") not in registered:
                            errors.append(f"unregistered Appendix diagnostics row hash in {rel}")
                        for key in ("baseline_config_hash", "refined_config_hash"):
                            if row.get(key) and row.get(key) not in registered:
                                errors.append(f"unregistered Appendix diagnostics {key} in {rel}")
        except Exception as exc:
            errors.append(f"Appendix diagnostics registry validation failed: {exc}")
    residual_path = project_root / "data/processed/appendix_d5_ansatz_scaling.csv"
    if residual_path.exists():
        _, rows = _csv_rows(residual_path)
        fits = [r for r in rows if r.get("record_type") == "fit"]
        if {r.get("quantity") for r in fits} != {"R_ans", "E_phi"}:
            errors.append("residual scaling fit rows are incomplete")
        for quantity in ("R_ans", "E_phi"):
            sample_rows = [r for r in rows if r.get("record_type") == "sample" and r.get("quantity") == quantity]
            if len(sample_rows) != 4:
                errors.append(f"residual scaling needs four sample rows for {quantity}")
        for row in fits:
            for key in ("slope", "intercept", "R_squared"):
                try:
                    if not np.isfinite(float(row[key])):
                        errors.append(f"nonfinite {key} in residual scaling")
                except Exception:
                    errors.append(f"invalid {key} in residual scaling")
    if require_local_records:
        validation_path = project_root / "data/processed/validation_report_appendix_d5.json"
        if validation_path.exists():
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                if validation.get("status") != "passed":
                    errors.append("Appendix validation report status is not passed")
            except Exception as exc:
                errors.append(f"Appendix validation report could not be read: {exc}")
        migration_path = project_root / "data/processed/appendix_migration_equality_report.json"
        if migration_path.exists():
            try:
                migration = json.loads(migration_path.read_text(encoding="utf-8"))
                if migration.get("status") != "passed":
                    errors.append("Appendix migration equality report status is not passed")
            except Exception as exc:
                errors.append(f"Appendix migration equality report could not be read: {exc}")
        manifest = project_root / "data/processed/release_manifest_d5.txt"
        if manifest.exists():
            text = manifest.read_text(encoding="utf-8")
            if "Appendix diagnostics status:" not in text:
                errors.append("release manifest has no Appendix diagnostics run record")
        else:
            errors.append("release manifest missing")
        failures_path = project_root / "data/processed/failures_exp06_d5_block_selection.json"
        failure_errors, failure_warnings = _validate_failure_report(failures_path)
        errors.extend(failure_errors)
        warnings.extend(failure_warnings)
        try:
            source_fingerprint = compute_source_fingerprint(project_root)
        except Exception as exc:
            source_fingerprint = None
            errors.append(f"could not compute source fingerprint: {exc}")
        receipt_errors, receipt_warnings = validate_receipt(
            project_root / "data/processed/run_receipt_appendix_d5.json",
            phase="appendix_d5",
            project_root=project_root,
            expected_source_fingerprint=source_fingerprint,
            expected_precision_mode="x64",
            expected_jax_x64=True,
            expected_phase_a_source_config_hash=phase_a_hash,
            expected_appendix_package_config_hash=appendix_d5_package_hash,
            required_argv_tokens=("--run", "all", "--x64"),
            required_artifacts=(
                *APPENDIX_D5_REQUIRED_FILES,
            ),
        )
        errors.extend(receipt_errors)
        warnings.extend(receipt_warnings)
    return {
        "status": "failed" if errors else "completed with warnings" if warnings else "passed",
        "errors": errors,
        "warnings": warnings,
        "checked_artifacts": checked,
    }


def read_recorded_regression_status(project_root: Path = PROJECT_ROOT) -> str:
    """Return the recorded immutable Figure 1--4 regression status."""
    errors, _ = validate_receipt(
        project_root / "data/processed/run_receipt_regression_fig1_4.json",
        phase="figures_1_4_regression",
        project_root=project_root,
        expected_config_hash="figures_1_4_cache_first",
        required_artifacts=("data/processed/regression_fig1_4_report.json",),
    )
    if errors:
        return "failed" if (project_root / "data/processed/run_receipt_regression_fig1_4.json").exists() else "not run"
    report_path = project_root / "data/processed/regression_fig1_4_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        receipt = json.loads((project_root / "data/processed/run_receipt_regression_fig1_4.json").read_text(encoding="utf-8"))
    except Exception:
        return "failed"
    report_bytes = report_path.read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    if receipt.get("regression_report_path") != "data/processed/regression_fig1_4_report.json":
        return "failed"
    if receipt.get("regression_report_sha256") != report_hash:
        return "failed"
    if receipt.get("regression_report_schema_version") != report.get("schema_version"):
        return "failed"
    if report.get("status") != "passed":
        return "failed"
    if report.get("mapping_status") != "passed":
        return "failed"
    if report.get("global_legacy_artifact_diff", {}).get("status") != "passed":
        return "failed"
    figures = report.get("figure_results", [])
    expected = {"Figure 1", "Figure 2", "Figure 3", "Figure 4"}
    if {item.get("manuscript_figure") for item in figures} != expected:
        return "failed"
    if any(item.get("status") != "passed" for item in figures):
        return "failed"
    return "passed"


def appendix_d5_artifacts_exist() -> bool:
    """Backward-compatible wrapper for content-aware Appendix diagnostics validation."""
    return validate_appendix_d5_completion(require_local_records=False)["status"] in {
        "passed",
        "completed with warnings",
    }


def publication_completion_status(project_root: Path = PROJECT_ROOT) -> str:
    public_phase_a = validate_phase_a_completion(project_root, require_local_records=False)
    public_appendix = validate_appendix_d5_completion(project_root, require_local_records=False)
    if (
        public_phase_a["status"] not in {"passed", "completed with warnings"}
        or public_appendix["status"] not in {"passed", "completed with warnings"}
    ):
        return "Public Figure 6 or Appendix artifacts failed validation"

    full_phase_a = validate_phase_a_completion(project_root, require_local_records=True)
    full_appendix = validate_appendix_d5_completion(project_root, require_local_records=True)
    regression = read_recorded_regression_status(project_root)
    if (
        full_phase_a["status"] in {"passed", "completed with warnings"}
        and full_appendix["status"] in {"passed", "completed with warnings"}
        and regression == "passed"
    ):
        return "Full publication package completed"
    return "Core public Figure 6 and Appendix artifacts completed; local validation records are optional"


def aggregate_status(statuses: Sequence[str]) -> str:
    """Aggregate pass/warning/fail statuses."""
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warning" for status in statuses):
        return "warning"
    return "pass"


def _failure_record(args: argparse.Namespace, K: float, stage: str, exc: BaseException, safe: bool = True) -> Dict[str, Any]:
    return {
        "phase": "A",
        "K": float(K),
        "frequency_mode": args.frequency_mode,
        "seed": args.seed,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback_excerpt": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=5)),
        "safe_to_continue": bool(safe),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_case(
    args: argparse.Namespace,
    K: float,
    package: Mapping[str, Any],
) -> Dict[str, Any]:
    rho = float(package["rho_numerator"]) / float(K)
    ts = make_time_grid(K, args.tau_max)
    init = make_d5_initial_state(
        package["delta_matrices"],
        K,
        args.phi0,
        tuple(INITIAL_BLOCK_WEIGHTS),
        tuple(THETA_BLOCK_VALUES),
        rho,
        args.c_init,
    )
    result = integrate_d5(
        init["x0"],
        package["omega_matrices"],
        K,
        ts,
        rtol=args.rtol,
        atol=args.atol,
        dt0=args.dt0,
        max_steps=args.max_steps,
    )
    phi, u, sin_phi = cartesian_to_polar_d5(result["x"], eps_pole=EPS_POLE)
    min_sin_phi = np.min(sin_phi, axis=1)
    phi_bar = np.mean(phi, axis=1)
    E_phi = np.max(np.abs(phi - phi_bar[:, None]), axis=1)
    u_rot = rotate_transverse_frame(u, ts, package["bar_omega_empirical"])
    u_star, mean_norm = common_transverse_direction(u_rot)
    p_sim = block_weights(u_star)
    R_ans, R_unit = first_order_ansatz_residuals(u_rot, u_star, package["delta_matrices"], K)
    D_u = transverse_diameter(u_rot)
    lower_invalid = (min_sin_phi < 1e-6) | (mean_norm < MEAN_TRANSVERSE_INVALID) | (phi_bar < METRIC_PHI_MIN)
    invalid_indices = np.where(lower_invalid)[0]
    search_stop = int(invalid_indices[0] - 1) if invalid_indices.size else len(ts) - 1
    t_f, idx_f, reached = find_persistent_fast_time(
        ts,
        R_ans,
        E_phi,
        rho,
        K,
        args.C_tol,
        args.persistence_Kt,
        search_stop_index=search_stop,
    )
    fast_unresolved = not bool(reached)
    p_red = np.full_like(p_sim, np.nan)
    phi_red = np.full_like(phi_bar, np.nan)
    lambda_red = np.full_like(phi_bar, np.nan)
    if reached:
        p_red = reduced_block_weights(ts, t_f, p_sim[idx_f], package["variance_empirical"], K)
        phi_red = reduced_mean_polar_angle(ts, t_f, phi_bar[idx_f], p_sim[idx_f], package["variance_empirical"], K)
        lambda_red = np.sum(p_red * package["variance_empirical"][None, :], axis=1) / float(K)
    lambda_sim = np.sum(p_sim * package["variance_empirical"][None, :], axis=1) / float(K)
    D_exact = exact_mean_polar_drift_coefficient(phi, u_rot, K, phi_bar, eps_pole=EPS_POLE)

    if reached:
        post_fast_valid = (
            (ts >= t_f)
            & (mean_norm >= MEAN_TRANSVERSE_WARNING)
            & (min_sin_phi >= 1e-6)
            & (mean_norm >= MEAN_TRANSVERSE_INVALID)
        )
        polar_metric_valid = (
            post_fast_valid
            & (phi_bar >= METRIC_PHI_MIN)
            & (phi_bar <= METRIC_PHI_MAX)
            & np.isfinite(D_exact)
        )
    else:
        post_fast_valid = np.zeros_like(ts, dtype=bool)
        polar_metric_valid = np.zeros_like(ts, dtype=bool)

    slope_fit = fit_log_weight_ratio_slope(ts, p_sim, t_f, K, package["variance_empirical"], post_fast_valid)
    p_rel = relative_l2(p_sim, p_red, post_fast_valid[:, None])
    p_max = _safe_max_abs((p_sim - p_red)[post_fast_valid])
    drift_rel = relative_l2(D_exact, lambda_red, polar_metric_valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift_max_rel = _safe_max_abs((D_exact - lambda_red)[polar_metric_valid] / D_exact[polar_metric_valid])
        Y_sim = np.log(np.tan(phi_bar) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
        Y_red = np.log(np.tan(phi_red) / np.tan(phi_bar[idx_f])) if reached else np.full_like(phi_bar, np.nan)
    polar_rel = relative_l2(Y_sim, Y_red, polar_metric_valid)
    max_phi_error = _safe_max_abs((phi_bar - phi_red)[polar_metric_valid])
    post_count = int(np.sum(post_fast_valid))
    polar_count = int(np.sum(polar_metric_valid))
    min_mean_post = float(np.nanmin(mean_norm[ts >= t_f])) if reached else float("nan")
    min_sin_post = float(np.nanmin(min_sin_phi[ts >= t_f])) if reached else float("nan")
    block_sum_error = _safe_max_abs(np.sum(p_sim, axis=1) - 1.0)

    row = {
        "K": float(K),
        "rho": rho,
        "t_f_num": float(t_f),
        "K_t_f_num": float(K * t_f) if reached else float("nan"),
        "fast_threshold_reached": bool(reached),
        "fast_layer_not_resolved": bool(fast_unresolved),
        "median_R_ans_over_rho2": finite_percentile(R_ans[post_fast_valid] / rho**2, 50),
        "p95_R_ans_over_rho2": finite_percentile(R_ans[post_fast_valid] / rho**2, 95),
        "median_E_phi_over_rho2": finite_percentile(E_phi[post_fast_valid] / rho**2, 50),
        "p95_E_phi_over_rho2": finite_percentile(E_phi[post_fast_valid] / rho**2, 95),
        "D_u_over_rho_median": finite_percentile(D_u[post_fast_valid] / rho, 50),
        "log_ratio_slope_fit": slope_fit["slope"],
        "log_ratio_slope_pred": slope_fit["slope_pred"],
        "log_ratio_slope_relative_error": slope_fit["relative_error"],
        "log_ratio_r_squared": slope_fit["r_squared"],
        "p_relative_L2_error": p_rel,
        "p_max_error": p_max,
        "drift_relative_L2_error": drift_rel,
        "drift_max_relative_error": drift_max_rel,
        "polar_logtan_relative_L2_error": polar_rel,
        "max_phi_error": max_phi_error,
        "sphere_norm_error": float(result["sphere_error"]),
        "min_mean_transverse_norm": min_mean_post,
        "max_block_weight_sum_error": block_sum_error,
        "post_fast_metric_point_count": post_count,
        "polar_metric_point_count": polar_count,
        "warning_count": 0,
    }
    row["warning_count"] = _row_warning_count(row)
    assumption = {
        "K": float(K),
        "rho": rho,
        "median_R_ans_over_rho2": row["median_R_ans_over_rho2"],
        "p95_R_ans_over_rho2": row["p95_R_ans_over_rho2"],
        "median_E_phi_over_rho2": row["median_E_phi_over_rho2"],
        "p95_E_phi_over_rho2": row["p95_E_phi_over_rho2"],
        "median_D_u_over_rho": finite_percentile(D_u[post_fast_valid] / rho, 50),
        "p95_D_u_over_rho": finite_percentile(D_u[post_fast_valid] / rho, 95),
        "min_mean_transverse_norm": min_mean_post,
        "min_sin_phi": min_sin_post,
    }

    return {
        "ts": ts,
        "T_end": float(ts[-1]),
        "rho": rho,
        "initial_state": init["x0"],
        "phi_bar": phi_bar,
        "phi_red": phi_red,
        "E_phi": E_phi,
        "R_ans": R_ans,
        "R_unit": R_unit,
        "D_u": D_u,
        "mean_transverse_norm": mean_norm,
        "min_sin_phi": min_sin_phi,
        "p_sim": p_sim,
        "p_red": p_red,
        "D_exact": D_exact,
        "lambda_sim": lambda_sim,
        "lambda_red": lambda_red,
        "t_f_num": float(t_f),
        "t_f_index": int(idx_f),
        "fast_threshold_reached": bool(reached),
        "fast_layer_not_resolved": bool(fast_unresolved),
        "sphere_norm_error": float(result["sphere_error"]),
        "post_fast_valid_mask": post_fast_valid,
        "polar_metric_valid_mask": polar_metric_valid,
        "summary_row": row,
        "assumption_row": assumption,
    }


def recompute(args: argparse.Namespace, config_dict: Mapping[str, Any], conf_hash: str) -> Dict[str, Any]:
    package = make_two_block_frequency_package(
        N=args.N,
        sigma_std_target=tuple(args.sigma_std_target),
        bar_omega_target=tuple(args.bar_omega_values),
        mode=args.frequency_mode,
        seed=args.seed,
    )
    structure = validate_common_axis_structure(package["omega_matrices"])
    cases: List[Dict[str, Any] | None] = []
    failures: List[Dict[str, Any]] = []
    for K in args.k_values:
        print(f"Running d=5 case K={K:g}")
        try:
            cases.append(run_case(args, float(K), package))
        except Exception as exc:
            failures.append(_failure_record(args, float(K), "case", exc, safe=True))
            cases.append(None)
            print(f"Failed K={K:g}: {exc}")

    nK = len(args.k_values)
    Tmax = max(len(c["ts"]) for c in cases if c is not None) if any(c is not None for c in cases) else 0
    if Tmax == 0:
        raise RuntimeError("all d=5 cases failed; no publication cache can be generated.")
    N = int(args.N)

    def arr2(shape: Sequence[int], fill=np.nan, dtype=float):
        return np.full(tuple(shape), fill, dtype=dtype)

    data: Dict[str, Any] = {
        "schema_version": np.array(SCHEMA_VERSION),
        "perturbation_schema_version": np.array(PERTURBATION_SCHEMA_VERSION),
        "config_hash": np.array(conf_hash),
        "K_values": np.asarray(args.k_values, dtype=float),
        "rho_values": arr2((nK,)),
        "T_end_values": arr2((nK,)),
        "ts": arr2((nK, Tmax)),
        "valid_time_mask": np.zeros((nK, Tmax), dtype=bool),
        "frequency_block1": package["omega_blocks"][:, 0],
        "frequency_block2": package["omega_blocks"][:, 1],
        "delta_block1": package["delta_blocks"][:, 0],
        "delta_block2": package["delta_blocks"][:, 1],
        "sigma_std_target": package["sigma_std_target"],
        "sigma_std_empirical": package["sigma_std_empirical"],
        "variance_empirical": package["variance_empirical"],
        "bar_omega_target": package["bar_omega_target"],
        "bar_omega_empirical": package["bar_omega_empirical"],
        "Q_transverse": package["Q_transverse"],
        "sample_correlation": np.array(package["sample_correlation"]),
        "orthogonalized": np.array(package["orthogonalized"], dtype=bool),
        "max_skew_residual": np.array(structure["max_skew_residual"]),
        "max_axis_residual": np.array(structure["max_axis_residual"]),
        "max_commutator_residual": np.array(structure["max_commutator_residual"]),
        "Q_block_error": np.array(package["Q_block_error"]),
        "initial_block_weights": np.tile(np.asarray(INITIAL_BLOCK_WEIGHTS, dtype=float), (nK, 1)),
        "initial_states": arr2((nK, N, 5)),
        "phi_bar": arr2((nK, Tmax)),
        "phi_red": arr2((nK, Tmax)),
        "E_phi": arr2((nK, Tmax)),
        "R_ans": arr2((nK, Tmax)),
        "R_unit": arr2((nK, Tmax)),
        "D_u": arr2((nK, Tmax)),
        "mean_transverse_norm": arr2((nK, Tmax)),
        "min_sin_phi": arr2((nK, Tmax)),
        "p_sim": arr2((nK, Tmax, 2)),
        "p_red": arr2((nK, Tmax, 2)),
        "D_exact": arr2((nK, Tmax)),
        "lambda_sim": arr2((nK, Tmax)),
        "lambda_red": arr2((nK, Tmax)),
        "t_f_num": arr2((nK,)),
        "t_f_index": np.full((nK,), -1, dtype=int),
        "fast_threshold_reached": np.zeros((nK,), dtype=bool),
        "fast_layer_not_resolved": np.ones((nK,), dtype=bool),
        "sphere_norm_error": arr2((nK,)),
        "post_fast_valid_mask": np.zeros((nK, Tmax), dtype=bool),
        "polar_metric_valid_mask": np.zeros((nK, Tmax), dtype=bool),
        "precision_mode": np.array("x64"),
    }
    summary_rows: List[Dict[str, Any]] = []
    assumption_rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        if case is None:
            continue
        T = len(case["ts"])
        data["rho_values"][i] = case["rho"]
        data["T_end_values"][i] = case["T_end"]
        data["ts"][i, :T] = case["ts"]
        data["valid_time_mask"][i, :T] = True
        data["initial_states"][i] = case["initial_state"]
        for key in [
            "phi_bar",
            "phi_red",
            "E_phi",
            "R_ans",
            "R_unit",
            "D_u",
            "mean_transverse_norm",
            "min_sin_phi",
            "D_exact",
            "lambda_sim",
            "lambda_red",
            "post_fast_valid_mask",
            "polar_metric_valid_mask",
        ]:
            data[key][i, :T] = case[key]
        data["p_sim"][i, :T, :] = case["p_sim"]
        data["p_red"][i, :T, :] = case["p_red"]
        data["t_f_num"][i] = case["t_f_num"]
        data["t_f_index"][i] = case["t_f_index"]
        data["fast_threshold_reached"][i] = case["fast_threshold_reached"]
        data["fast_layer_not_resolved"][i] = case["fast_layer_not_resolved"]
        data["sphere_norm_error"][i] = case["sphere_norm_error"]
        summary_rows.append(case["summary_row"])
        assumption_rows.append(case["assumption_row"])
        if case["fast_layer_not_resolved"]:
            failures.append(
                {
                    "phase": "A",
                    "K": float(args.k_values[i]),
                    "frequency_mode": args.frequency_mode,
                    "seed": args.seed,
                    "stage": "fast_layer",
                    "exception_type": "ScientificTargetNotReached",
                    "message": "persistent fast threshold was not reached on the permitted search interval.",
                    "traceback_excerpt": "",
                    "safe_to_continue": True,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
    data["_summary_rows"] = summary_rows
    data["_assumption_rows"] = assumption_rows
    data["_failures"] = failures
    data["_config_dict"] = dict(config_dict)
    return data


def validate_cache(data: Mapping[str, Any], expected_hash: str) -> None:
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in data]
    if missing:
        raise RuntimeError(f"Figure 6 cache is missing required keys: {missing}")
    schema = str(np.asarray(data["schema_version"]).item())
    perturbation = str(np.asarray(data["perturbation_schema_version"]).item())
    found_hash = str(np.asarray(data["config_hash"]).item())
    precision = str(np.asarray(data["precision_mode"]).item())
    errors = []
    if schema != SCHEMA_VERSION:
        errors.append(f"schema_version {schema!r} != {SCHEMA_VERSION!r}")
    if perturbation != PERTURBATION_SCHEMA_VERSION:
        errors.append(f"perturbation_schema_version {perturbation!r} != {PERTURBATION_SCHEMA_VERSION!r}")
    if found_hash != expected_hash:
        errors.append(f"config_hash {found_hash} != {expected_hash}")
    if precision != "x64":
        errors.append(f"precision_mode {precision!r} is not publication x64")
    K_values = np.asarray(data["K_values"])
    nK = len(K_values)
    N = len(np.asarray(data["frequency_block1"]))
    Tmax = np.asarray(data["ts"]).shape[1]
    expected_shapes = {
        "rho_values": (nK,),
        "ts": (nK, Tmax),
        "valid_time_mask": (nK, Tmax),
        "frequency_block1": (N,),
        "frequency_block2": (N,),
        "delta_block1": (N,),
        "delta_block2": (N,),
        "sigma_std_target": (2,),
        "sigma_std_empirical": (2,),
        "variance_empirical": (2,),
        "bar_omega_target": (2,),
        "bar_omega_empirical": (2,),
        "Q_transverse": (4, 4),
        "initial_states": (nK, N, 5),
        "p_sim": (nK, Tmax, 2),
        "p_red": (nK, Tmax, 2),
        "post_fast_valid_mask": (nK, Tmax),
        "polar_metric_valid_mask": (nK, Tmax),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(data[key]).shape != shape:
            errors.append(f"{key} shape {np.asarray(data[key]).shape} != {shape}")
    if errors:
        raise RuntimeError("stale or invalid Figure 6 cache:\n" + "\n".join(errors))


def load_cache(cache_path: Path, expected_hash: str) -> Dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    validate_cache(data, expected_hash)
    return data


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rows_from_cache(data: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    assumption_rows: List[Dict[str, Any]] = []
    for i, K in enumerate(np.asarray(data["K_values"], dtype=float)):
        valid = np.asarray(data["post_fast_valid_mask"][i], dtype=bool)
        polar = np.asarray(data["polar_metric_valid_mask"][i], dtype=bool)
        rho = float(data["rho_values"][i])
        p_sim = np.asarray(data["p_sim"][i])
        p_red = np.asarray(data["p_red"][i])
        slope_fit = fit_log_weight_ratio_slope(
            data["ts"][i],
            p_sim,
            float(data["t_f_num"][i]),
            K,
            data["variance_empirical"],
            valid,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            Y_sim = np.log(np.tan(data["phi_bar"][i]) / np.tan(data["phi_bar"][i, int(data["t_f_index"][i])]))
            Y_red = np.log(np.tan(data["phi_red"][i]) / np.tan(data["phi_bar"][i, int(data["t_f_index"][i])]))
            drift_rel_vec = (data["D_exact"][i] - data["lambda_red"][i]) / data["D_exact"][i]
        row = {
            "K": float(K),
            "rho": rho,
            "t_f_num": float(data["t_f_num"][i]),
            "K_t_f_num": float(K * data["t_f_num"][i]) if data["fast_threshold_reached"][i] else float("nan"),
            "fast_threshold_reached": bool(data["fast_threshold_reached"][i]),
            "fast_layer_not_resolved": bool(data["fast_layer_not_resolved"][i]),
            "median_R_ans_over_rho2": finite_percentile(data["R_ans"][i, valid] / rho**2, 50),
            "p95_R_ans_over_rho2": finite_percentile(data["R_ans"][i, valid] / rho**2, 95),
            "median_E_phi_over_rho2": finite_percentile(data["E_phi"][i, valid] / rho**2, 50),
            "p95_E_phi_over_rho2": finite_percentile(data["E_phi"][i, valid] / rho**2, 95),
            "D_u_over_rho_median": finite_percentile(data["D_u"][i, valid] / rho, 50),
            "log_ratio_slope_fit": slope_fit["slope"],
            "log_ratio_slope_pred": slope_fit["slope_pred"],
            "log_ratio_slope_relative_error": slope_fit["relative_error"],
            "log_ratio_r_squared": slope_fit["r_squared"],
            "p_relative_L2_error": relative_l2(p_sim, p_red, valid[:, None]),
            "p_max_error": _safe_max_abs((p_sim - p_red)[valid]),
            "drift_relative_L2_error": relative_l2(data["D_exact"][i], data["lambda_red"][i], polar),
            "drift_max_relative_error": _safe_max_abs(drift_rel_vec[polar]),
            "polar_logtan_relative_L2_error": relative_l2(Y_sim, Y_red, polar),
            "max_phi_error": _safe_max_abs((data["phi_bar"][i] - data["phi_red"][i])[polar]),
            "sphere_norm_error": float(data["sphere_norm_error"][i]),
            "min_mean_transverse_norm": finite_percentile(data["mean_transverse_norm"][i, valid], 0),
            "max_block_weight_sum_error": _safe_max_abs(np.sum(p_sim, axis=1) - 1.0),
            "post_fast_metric_point_count": int(np.sum(valid)),
            "polar_metric_point_count": int(np.sum(polar)),
            "warning_count": 0,
        }
        row["warning_count"] = _row_warning_count(row)
        rows.append(row)
        tmask = data["ts"][i] >= data["t_f_num"][i] if data["fast_threshold_reached"][i] else np.zeros_like(data["ts"][i], dtype=bool)
        assumption_rows.append(
            {
                "K": float(K),
                "rho": rho,
                "median_R_ans_over_rho2": row["median_R_ans_over_rho2"],
                "p95_R_ans_over_rho2": row["p95_R_ans_over_rho2"],
                "median_E_phi_over_rho2": row["median_E_phi_over_rho2"],
                "p95_E_phi_over_rho2": row["p95_E_phi_over_rho2"],
                "median_D_u_over_rho": finite_percentile(data["D_u"][i, valid] / rho, 50),
                "p95_D_u_over_rho": finite_percentile(data["D_u"][i, valid] / rho, 95),
                "min_mean_transverse_norm": finite_percentile(data["mean_transverse_norm"][i, tmask], 0),
                "min_sin_phi": finite_percentile(data["min_sin_phi"][i, tmask], 0),
            }
        )
    return rows, assumption_rows


def fit_loglog_scaling(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit log(y) = slope*log(x) + intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.sum(mask)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "R_squared": float("nan")}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "R_squared": float(r2)}


def fast_threshold_row_for_case(data: Mapping[str, Any], i: int) -> Dict[str, Any]:
    """Build one fast-layer diagnostic row from the Phase A cache."""
    K = float(data["K_values"][i])
    rho = float(data["rho_values"][i])
    valid = np.asarray(data["valid_time_mask"][i], dtype=bool)
    ts = np.asarray(data["ts"][i, valid], dtype=float)
    R = np.asarray(data["R_ans"][i, valid], dtype=float)
    E = np.asarray(data["E_phi"][i, valid], dtype=float)
    mean_norm = np.asarray(data["mean_transverse_norm"][i, valid], dtype=float)
    min_sin = np.asarray(data["min_sin_phi"][i, valid], dtype=float)
    phi_bar = np.asarray(data["phi_bar"][i, valid], dtype=float)
    invalid = (min_sin < 1e-6) | (mean_norm < MEAN_TRANSVERSE_INVALID) | (phi_bar < METRIC_PHI_MIN)
    invalid_idx = np.where(invalid)[0]
    search_stop = int(invalid_idx[0] - 1) if invalid_idx.size else len(ts) - 1
    t_eval = 10.0 / K
    row: Dict[str, Any] = {
        "K": K,
        "rho": rho,
        "initial_R_ans": float(R[0]),
        "initial_E_phi": float(E[0]),
        "initial_R_ans_over_rho": float(R[0] / rho),
        "initial_E_phi_over_rho": float(E[0] / rho),
        "initial_R_ans_over_rho2": float(R[0] / rho**2),
        "initial_E_phi_over_rho2": float(E[0] / rho**2),
        "R_ans_at_Kt10": float(np.interp(t_eval, ts, R)),
        "E_phi_at_Kt10": float(np.interp(t_eval, ts, E)),
    }
    row["R_ans_at_Kt10_over_rho2"] = float(row["R_ans_at_Kt10"] / rho**2)
    row["E_phi_at_Kt10_over_rho2"] = float(row["E_phi_at_Kt10"] / rho**2)
    threshold_values: Dict[str, tuple[float, bool]] = {}
    for C_tol in (1.0, 3.0, 5.0, 8.0):
        t_f, _, reached = find_persistent_fast_time(
            ts, R, E, rho, K, C_tol, 2.0, search_stop_index=search_stop
        )
        tag = str(int(C_tol))
        threshold_values[tag] = (float(K * t_f) if reached else float("nan"), bool(reached))
    for tag in ("1", "3", "5", "8"):
        row[f"Kt_f_Ctol_{tag}"] = threshold_values[tag][0]
    for tag in ("1", "3", "5", "8"):
        row[f"fast_threshold_reached_Ctol_{tag}"] = threshold_values[tag][1]
    return row


def build_fast_layer_diagnostics(data: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Return diagnostic rows and scaling fits for the fast-layer report."""
    rows = [fast_threshold_row_for_case(data, i) for i in range(len(data["K_values"]))]
    rho = np.asarray([row["rho"] for row in rows], dtype=float)
    fits = {
        "initial_R_ans": fit_loglog_scaling(rho, np.asarray([row["initial_R_ans"] for row in rows])),
        "initial_E_phi": fit_loglog_scaling(rho, np.asarray([row["initial_E_phi"] for row in rows])),
        "R_ans_at_Kt10": fit_loglog_scaling(rho, np.asarray([row["R_ans_at_Kt10"] for row in rows])),
        "E_phi_at_Kt10": fit_loglog_scaling(rho, np.asarray([row["E_phi_at_Kt10"] for row in rows])),
    }
    return rows, fits


def environment_metadata() -> Dict[str, Any]:
    try:
        diffrax_version = importlib.import_module("diffrax").__version__
    except Exception:
        diffrax_version = "unavailable"
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "diffrax_version": diffrax_version,
        "matplotlib_version": plt.matplotlib.__version__,
    }


def make_metadata(
    args: argparse.Namespace,
    data: Mapping[str, Any],
    config_dict: Mapping[str, Any],
    conf_hash: str,
    git_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = {
        "experiment": "d5_block_selection",
        "schema_version": SCHEMA_VERSION,
        "perturbation_schema_version": PERTURBATION_SCHEMA_VERSION,
        "manuscript_equations": [19, 21, 22, 25, 26, 27, 28],
        "dimension": 5,
        "N": int(args.N),
        "K_values": np.asarray(data["K_values"], dtype=float),
        "sigma_std_target": np.asarray(data["sigma_std_target"], dtype=float),
        "sigma_std_empirical": np.asarray(data["sigma_std_empirical"], dtype=float),
        "variance_empirical": np.asarray(data["variance_empirical"], dtype=float),
        "bar_omega_target": np.asarray(data["bar_omega_target"], dtype=float),
        "bar_omega_empirical": np.asarray(data["bar_omega_empirical"], dtype=float),
        "frequency_mode": args.frequency_mode,
        "seed": args.seed,
        "sample_correlation": float(data["sample_correlation"]),
        "orthogonalized": bool(data["orthogonalized"]),
        "phi0": float(args.phi0),
        "initial_block_weights": INITIAL_BLOCK_WEIGHTS,
        "theta_block_values": THETA_BLOCK_VALUES,
        "c_init": float(args.c_init),
        "C_tol": float(args.C_tol),
        "persistence_Kt": float(args.persistence_Kt),
        "tau_max": float(args.tau_max),
        "metric_phi_min": METRIC_PHI_MIN,
        "metric_phi_max": METRIC_PHI_MAX,
        "mean_transverse_warning": MEAN_TRANSVERSE_WARNING,
        "mean_transverse_invalid": MEAN_TRANSVERSE_INVALID,
        "eps_pole": EPS_POLE,
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "dt0": float(args.dt0),
        "max_steps": int(args.max_steps),
        "jax_x64": bool(jax.config.jax_enable_x64),
        "git_commit": git_metadata.get("git_commit", "unavailable"),
        "git_dirty": git_metadata.get("git_dirty"),
        "config_hash": conf_hash,
        "source_fingerprint": compute_source_fingerprint(PROJECT_ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "max_skew_residual": float(data["max_skew_residual"]),
        "max_axis_residual": float(data["max_axis_residual"]),
        "max_commutator_residual": float(data["max_commutator_residual"]),
        "Q_block_error": float(data["Q_block_error"]),
        "configuration": dict(config_dict),
    }
    metadata.update(environment_metadata())
    return metadata


def write_text_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    assumption_rows: Sequence[Mapping[str, Any]],
    fast_rows: Sequence[Mapping[str, Any]],
    fast_fits: Mapping[str, Mapping[str, float]],
    failures: Sequence[Mapping[str, Any]],
    conf_hash: str,
    git_metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Five-dimensional block-selection summary",
        "=========================================",
        "",
        f"schema_version: {SCHEMA_VERSION}",
        f"perturbation_schema_version: {PERTURBATION_SCHEMA_VERSION}",
        f"config_hash: {conf_hash}",
        f"source_fingerprint: {compute_source_fingerprint(PROJECT_ROOT)}",
        f"git_commit: {git_metadata.get('git_commit', 'unavailable')}",
        f"git_dirty: {git_metadata.get('git_dirty')}",
        "",
        "Phase A status by K",
        "-------------------",
    ]
    for row in rows:
        lines.append(
            "K={K:g}, rho={rho:.6g}, t_f={t_f_num:.6g}, "
            "slope_rel_err={log_ratio_slope_relative_error:.6g}, "
            "polar_rel_L2={polar_logtan_relative_L2_error:.6g}, "
            "warnings={warning_count}".format(**row)
        )
    lines.extend(["", "Assumption diagnostics", "----------------------"])
    for row in assumption_rows:
        lines.append(
            "K={K:g}, median R_ans/rho^2={median_R_ans_over_rho2:.6g}, "
            "p95 R_ans/rho^2={p95_R_ans_over_rho2:.6g}, "
            "min mean transverse norm={min_mean_transverse_norm:.6g}".format(**row)
        )
    lines.extend(
        [
            "",
            "Fast-layer threshold diagnostics",
            "--------------------------------",
            "C_tol=5 remains the primary pre-analysis threshold. C_tol=1 is exploratory.",
            (
                "Under C_tol=5, t_f=0 for coupling values whose initial "
                "normalized residuals already satisfy the persistent criterion."
            ),
        ]
    )
    for row in fast_rows:
        lines.append(
            "K={K:g}, Kt_f(Ctol=1,3,5,8)=({Kt_f_Ctol_1:.6g}, "
            "{Kt_f_Ctol_3:.6g}, {Kt_f_Ctol_5:.6g}, {Kt_f_Ctol_8:.6g}), "
            "R_ans(0)/rho={initial_R_ans_over_rho:.6g}, "
            "R_ans(Kt=10)/rho^2={R_ans_at_Kt10_over_rho2:.6g}".format(**row)
        )
    lines.extend(["", "Fast-layer scaling fits", "-----------------------"])
    for key, fit in fast_fits.items():
        lines.append(
            f"{key}: slope={fit['slope']:.12g}, intercept={fit['intercept']:.12g}, "
            f"R_squared={fit['R_squared']:.12g}"
        )
    lines.extend(["", f"Failure records: {len(failures)}"])
    for failure in failures:
        lines.append(f"- K={failure.get('K')}: {failure.get('stage')} {failure.get('message')}")
    completion = publication_completion_status()
    lines.extend(["", f"Completion status: {completion}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_manifest(path: Path, conf_hash: str) -> None:
    appendix_d5_files = [
        "data/cache/appendix_d5_diagnostics.npz",
        "data/processed/appendix_d5_ansatz_scaling.csv",
        "data/processed/appendix_d5_vector_law.csv",
        "data/processed/appendix_d5_controls.csv",
        "data/processed/appendix_d5_sensitivity.csv",
        "data/processed/appendix_tableA1_d5_diagnostics.csv",
        "paper/appendix_tableA1_d5_diagnostics.tex",
        "data/processed/appendix_d5_config_registry.json",
        "data/processed/metadata_appendix_d5.json",
        "data/processed/validation_report_appendix_d5.json",
        "data/processed/appendix_migration_equality_report.json",
        "figures/appendix_figA1_d5_ansatz_validation.pdf",
        "figures/appendix_figA1_d5_ansatz_validation.png",
        "figures/appendix_figA1_d5_ansatz_validation.eps",
        "figures/appendix_figA2_d5_controls_robustness.pdf",
        "figures/appendix_figA2_d5_controls_robustness.png",
        "figures/appendix_figA2_d5_controls_robustness.eps",
    ]
    public_phase_a_validation = validate_phase_a_completion(
        expected_config_hash=conf_hash,
        require_local_records=False,
    )
    public_appendix_d5_validation = validate_appendix_d5_completion(require_local_records=False)
    phase_a_validation = validate_phase_a_completion(expected_config_hash=conf_hash)
    appendix_d5_validation = validate_appendix_d5_completion()
    appendix_d5_status = appendix_d5_validation["status"]
    phase_a_status = phase_a_validation["status"]
    public_phase_a_status = public_phase_a_validation["status"]
    public_appendix_d5_status = public_appendix_d5_validation["status"]
    regression_status = read_recorded_regression_status()
    source_fingerprint = compute_source_fingerprint(PROJECT_ROOT)
    lines = [
        "D5 reproducibility manifest",
        "===========================",
        "",
        "Publication cache paths",
        "- data/cache/fig06_d5_block_selection.npz",
        "",
        "Processed data paths",
        "- data/processed/metadata_exp06_d5_block_selection.json",
        "- data/processed/summary_exp06_d5_block_selection.csv",
        "- data/processed/d5_block_selection_summary.txt",
        "- data/processed/failures_exp06_d5_block_selection.json",
        "- data/processed/run_receipt_phase_a_recompute.json",
        "- data/processed/run_receipt_phase_a_cache_render.json",
        "- data/processed/run_receipt_appendix_d5.json",
        "- data/processed/run_receipt_regression_fig1_4.json",
        "- data/processed/regression_fig1_4_report.json",
        "",
        "Figure paths",
        "- figures/fig6_d5_block_selection.pdf",
        "- figures/fig6_d5_block_selection.png",
        "- figures/fig6_d5_block_selection.eps",
        "",
        f"Config hash: {conf_hash}",
        f"Source fingerprint: {source_fingerprint}",
        "",
        "Exact reproduction commands",
        "- python -m pytest -q tests/test_high_dimensional_utils.py",
        "- python scripts/fig06_d5_block_selection.py --recompute --x64",
        "- python scripts/fig06_d5_block_selection.py",
        "- python scripts/appendix_d5_diagnostics.py --run all --x64",
        "",
        f"Public Figure 6 artifact status: {public_phase_a_status}",
        f"Public Appendix artifact status: {public_appendix_d5_status}",
        f"Full local Phase A status: {phase_a_status}",
        f"Full local Appendix diagnostics status: {appendix_d5_status}",
        f"Figures 1-4 regression status: {regression_status}",
        f"Phase A recompute receipt path: data/processed/run_receipt_phase_a_recompute.json",
        f"Phase A cache-render receipt path: data/processed/run_receipt_phase_a_cache_render.json",
        f"Appendix diagnostics receipt path: data/processed/run_receipt_appendix_d5.json",
        f"Figure 1-4 regression receipt path: data/processed/run_receipt_regression_fig1_4.json",
        f"Figure 1-4 regression report path: data/processed/regression_fig1_4_report.json",
    ]
    if phase_a_validation["errors"] or phase_a_validation["warnings"]:
        lines.extend(["", "Phase A validation details"])
        lines.extend([f"- error: {item}" for item in phase_a_validation["errors"]])
        lines.extend([f"- warning: {item}" for item in phase_a_validation["warnings"]])
    if appendix_d5_status in {"passed", "completed with warnings"}:
        lines.extend(["", "Appendix diagnostics files", *[f"- {item}" for item in appendix_d5_files]])
    if appendix_d5_validation["errors"] or appendix_d5_validation["warnings"]:
        lines.extend(["", "Appendix diagnostics validation details"])
        lines.extend([f"- error: {item}" for item in appendix_d5_validation["errors"]])
        lines.extend([f"- warning: {item}" for item in appendix_d5_validation["warnings"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_test_count_summary(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Return a static summary of the D5 validation test matrix."""
    test_path = project_root / "tests/test_high_dimensional_utils.py"
    summary: Dict[str, Any] = {
        "test_module": "tests/test_high_dimensional_utils.py",
        "static_test_functions": None,
        "cv_corruption_cases": None,
        "expected_pytest_collected": None,
        "required_cv_corruption_cases": 48,
    }
    if not test_path.exists():
        return summary
    text = test_path.read_text(encoding="utf-8")
    direct_tests = sum(1 for line in text.splitlines() if line.startswith("def test_"))
    cv_cases = text.count('("cv_')
    summary["static_test_functions"] = direct_tests
    summary["cv_corruption_cases"] = cv_cases
    summary["expected_pytest_collected"] = direct_tests + cv_cases
    return summary


def write_validation_report(path: Path, conf_hash: str) -> None:
    phase_a = validate_phase_a_completion(expected_config_hash=conf_hash)
    appendix_d5 = validate_appendix_d5_completion()
    regression_status = read_recorded_regression_status()
    appendix_d5_package_hash = None
    registry_path = PROJECT_ROOT / "data/processed/appendix_d5_config_registry.json"
    if registry_path.exists():
        try:
            appendix_d5_package_hash = json.loads(registry_path.read_text(encoding="utf-8")).get("appendix_package_config_hash")
        except Exception:
            appendix_d5_package_hash = None
    report = {
        "schema_version": "validation_report_d5_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_a_pipeline_status": phase_a.get("pipeline_status"),
        "phase_a_scientific_status": phase_a.get("scientific_status"),
        "phase_a_overall_status": phase_a.get("overall_status"),
        "phase_a_scientific_assessments": phase_a.get("scientific_assessments", []),
        "appendix_d5_status": appendix_d5.get("status"),
        "figure_1_4_regression_status": regression_status,
        "source_fingerprint": compute_source_fingerprint(PROJECT_ROOT),
        "phase_a_config_hash": conf_hash,
        "appendix_d5_package_hash": appendix_d5_package_hash,
        "receipt_validation": {
            "phase_a_errors": phase_a.get("errors", []),
            "appendix_d5_errors": appendix_d5.get("errors", []),
        },
        "artifact_integrity_status": "passed" if not phase_a.get("errors") and not appendix_d5.get("errors") else "failed",
        "test_counts": collect_test_count_summary(PROJECT_ROOT),
        "overall_completion_status": publication_completion_status(PROJECT_ROOT),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_figure(data: Mapping[str, Any], output_dir: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 17,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 11.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.55))
    axes = axes.ravel()
    K_values = np.asarray(data["K_values"], dtype=float)
    colors = color_cycle()
    rep_idx = int(np.argmin(np.abs(K_values - REPRESENTATIVE_K)))
    repK = float(K_values[rep_idx])
    rep_valid = np.asarray(data["valid_time_mask"][rep_idx], dtype=bool)
    ts = np.asarray(data["ts"][rep_idx, rep_valid], dtype=float)
    rho = float(data["rho_values"][rep_idx])
    R_norm = np.asarray(data["R_ans"][rep_idx, rep_valid], dtype=float) / rho**2
    E_norm = np.asarray(data["E_phi"][rep_idx, rep_valid], dtype=float) / rho**2
    axes[0].semilogy(repK * ts, R_norm, color=colors[1], linestyle="-", label=r"$R_{\rm ans}/\rho^2$")
    axes[0].semilogy(repK * ts, E_norm, color=colors[2], linestyle="--", label=r"$E_\phi/\rho^2$")
    axes[0].axhline(args.C_tol, color="0.25", linestyle=":", linewidth=1.3)
    if bool(data["fast_threshold_reached"][rep_idx]):
        ktf = repK * float(data["t_f_num"][rep_idx])
        if ktf > 1e-12:
            axes[0].axvline(ktf, color="0.1", linestyle="-.", linewidth=1.1)
            axes[0].axvspan(ktf, ktf + args.persistence_Kt, color="0.85", alpha=0.35, linewidth=0)
        else:
            axes[0].annotate(
                r"$Kt_f^{\rm num}=0$ for $C_{\rm tol}=5$",
                xy=(0.05, 0.08),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                fontsize=12.2,
            )
    axes[0].set(xlabel=r"$Kt$", ylabel=r"normalized error", xlim=(0, 12))
    axes[0].legend(loc="best")

    slope_ref = float(2.0 * (data["variance_empirical"][1] - data["variance_empirical"][0]))
    tau_ref = np.linspace(0.0, args.tau_max, 200)
    axes[1].plot(tau_ref, slope_ref * tau_ref, color="0.05", linewidth=1.8, linestyle="--", label="predicted slope")
    for i, K in enumerate(K_values):
        valid = np.asarray(data["post_fast_valid_mask"][i], dtype=bool)
        if not np.any(valid):
            continue
        tfi = float(data["t_f_num"][i])
        tau = (data["ts"][i, valid] - tfi) / float(K)
        ps = data["p_sim"][i, valid]
        y = np.log(ps[:, 0] / ps[:, 1])
        y = y - y[0]
        step = max(1, len(tau) // 10)
        axes[1].plot(
            tau,
            y,
            color=colors[i % len(colors)],
            linestyle=["-", "--", "-.", ":"][i % 4],
            marker=["o", "s", "^", "D"][i % 4],
            markevery=step,
            markersize=3.0,
            label=rf"$K={K:g}$",
        )
    axes[1].set(xlabel=r"$\tau$", ylabel=r"$Y_p(t)$")
    axes[1].legend(loc="best", ncol=1)

    polar = np.asarray(data["polar_metric_valid_mask"][rep_idx], dtype=bool)
    if np.any(polar):
        tau = (data["ts"][rep_idx, polar] - data["t_f_num"][rep_idx]) / repK
        step = max(1, int(np.sum(polar)) // 12)
        axes[2].plot(
            tau,
            repK * data["D_exact"][rep_idx, polar],
            color="0.0",
            linestyle="-",
            marker="o",
            markevery=step,
            markersize=3.0,
            linewidth=1.6,
            label=r"$KD_{\rm exact}$",
        )
        axes[2].plot(
            tau,
            repK * data["lambda_sim"][rep_idx, polar],
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label=r"$K\lambda_{\rm sim}$",
        )
        axes[2].plot(
            tau,
            repK * data["lambda_red"][rep_idx, polar],
            color="tab:orange",
            linestyle="-.",
            linewidth=1.5,
            label=r"$K\lambda_{\rm red}$",
        )
    axes[2].set(xlabel=r"$\tau$", ylabel=r"scaled drift coefficient")
    axes[2].legend(loc="best")

    for i, K in enumerate(K_values):
        polar = np.asarray(data["polar_metric_valid_mask"][i], dtype=bool)
        if not np.any(polar):
            continue
        idxf = int(data["t_f_index"][i])
        tau = (data["ts"][i, polar] - data["t_f_num"][i]) / float(K)
        y = np.log(np.tan(data["phi_bar"][i, polar]) / np.tan(data["phi_bar"][i, idxf]))
        yred = np.log(np.tan(data["phi_red"][i, polar]) / np.tan(data["phi_bar"][i, idxf]))
        axes[3].plot(tau, y, color=colors[i % len(colors)], linestyle=["-", "--", "-.", ":"][i % 4], label=rf"$K={K:g}$")
        if i == 0:
            axes[3].plot(tau, yred, color="0.05", linewidth=1.8, linestyle=(0, (5, 2)), label="reduced prediction")
        else:
            axes[3].plot(tau, yred, color="0.05", linewidth=0.9, linestyle=(0, (5, 2)), alpha=0.7)
    axes[3].set(xlabel=r"$\tau$", ylabel=r"$Y_\phi(t)$")
    handles, labels = axes[3].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = ["reduced prediction", r"$K=8$", r"$K=10$", r"$K=12$", r"$K=16$"]
    axes[3].legend(
        [handle_by_label[label] for label in ordered_labels if label in handle_by_label],
        [label for label in ordered_labels if label in handle_by_label],
        loc="upper right",
        ncol=1,
    )

    for ax in axes:
        format_axes(ax)
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes):
        ax.text(0.5, -0.25, label, transform=ax.transAxes, ha="center", va="top", fontsize=14.2)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.98, wspace=0.36, hspace=0.54)
    saved = save_figure_all_formats(fig, output_dir, "fig6_d5_block_selection")
    plt.close(fig)
    return saved


def strip_private(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if not key.startswith("_")}


def load_existing_failures(path: Path) -> List[Dict[str, Any]]:
    """Load an existing failure report if one is present."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _main_impl(argv: Sequence[str] | None = None) -> int:
    started_utc = utc_now()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    if args.x64 and args.float32_debug:
        raise SystemExit("--x64 and --float32-debug are mutually exclusive.")
    jax_config.update("jax_enable_x64", False if args.float32_debug else True)
    if args.float32_debug:
        print("DEBUG ONLY - NOT FOR PUBLICATION: float32 mode writes no publication artifacts.")
        return 0
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("publication Figure 6 run requires jax_enable_x64=True")

    config_dict = build_config(args)
    conf_hash = config_hash(config_dict)
    cache_path = _resolve_project_path(args.cache_path)
    figure_dir = _resolve_project_path(args.output_dir)
    processed_dir = _resolve_project_path(args.processed_dir)
    metadata_path = processed_dir / "metadata_exp06_d5_block_selection.json"
    summary_path = processed_dir / "summary_exp06_d5_block_selection.csv"
    report_path = processed_dir / "d5_block_selection_summary.txt"
    failure_path = processed_dir / "failures_exp06_d5_block_selection.json"
    manifest_path = processed_dir / "release_manifest_d5.txt"
    validation_report_path = processed_dir / "validation_report_d5.json"

    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute:
        data = recompute(args, config_dict, conf_hash)
        failures = data.pop("_failures")
        summary_rows = data.pop("_summary_rows")
        assumption_rows = data.pop("_assumption_rows")
        data.pop("_config_dict", None)
        np.savez_compressed(cache_path, **strip_private(data))
        print(f"Saved cache: {cache_path}")
    else:
        if not cache_path.exists():
            raise FileNotFoundError(f"missing Figure 6 cache {cache_path}; run with --recompute --x64 first.")
        data = load_cache(cache_path, conf_hash)
        summary_rows, assumption_rows = rows_from_cache(data)
        failures = load_existing_failures(failure_path)

    validate_cache(strip_private(data), conf_hash)
    fast_rows, fast_fits = build_fast_layer_diagnostics(data)
    git_metadata = get_git_metadata()
    metadata = make_metadata(args, data, config_dict, conf_hash, git_metadata)

    save_metadata(metadata_path, metadata)
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    if args.recompute or not failure_path.exists():
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    write_text_report(report_path, summary_rows, assumption_rows, fast_rows, fast_fits, failures, conf_hash, git_metadata)
    write_release_manifest(manifest_path, conf_hash)
    saved_paths = plot_figure(data, figure_dir, args)
    generated_artifacts = [cache_path] if args.recompute else [
        metadata_path,
        summary_path,
        failure_path,
        *saved_paths,
    ]
    write_run_receipt(
        processed_dir / ("run_receipt_phase_a_recompute.json" if args.recompute else "run_receipt_phase_a_cache_render.json"),
        phase="phase_a_recompute" if args.recompute else "phase_a_cache_render",
        command=["python", "scripts/fig06_d5_block_selection.py", *argv_list],
        argv=argv_list,
        started_utc=started_utc,
        return_code=0,
        status="completed with warnings" if failures else "passed",
        project_root=PROJECT_ROOT,
        config_hash=conf_hash,
        source_fingerprint=compute_source_fingerprint(PROJECT_ROOT),
        precision_mode="x64",
        jax_x64=bool(jax.config.jax_enable_x64),
        generated_artifacts=generated_artifacts,
        warnings=[str(failure.get("message", "")) for failure in failures if failure.get("safe_to_continue", False)],
        failures=[str(failure.get("message", "")) for failure in failures if not failure.get("safe_to_continue", False)],
    )
    write_text_report(report_path, summary_rows, assumption_rows, fast_rows, fast_fits, failures, conf_hash, git_metadata)
    write_release_manifest(manifest_path, conf_hash)
    write_validation_report(validation_report_path, conf_hash)

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved failure report: {failure_path}")
    print("Saved figure:")
    for path in saved_paths:
        print(f"  {path}")
    completion = publication_completion_status()
    print(f"Completion status: {completion}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    started_utc = utc_now()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit:
        raise
    conf_hash = config_hash(build_config(args)) if not args.float32_debug else ""
    jax_config.update("jax_enable_x64", False if args.float32_debug else True)
    processed_dir = _resolve_project_path(args.processed_dir)
    receipt_path = processed_dir / ("run_receipt_phase_a_recompute.json" if args.recompute else "run_receipt_phase_a_cache_render.json")
    phase = "phase_a_recompute" if args.recompute else "phase_a_cache_render"
    if not args.float32_debug:
        write_run_receipt(
            receipt_path,
            phase=phase,
            command=["python", "scripts/fig06_d5_block_selection.py", *argv_list],
            argv=argv_list,
            started_utc=started_utc,
            return_code=2,
            status="running",
            project_root=PROJECT_ROOT,
            config_hash=conf_hash,
            source_fingerprint=compute_source_fingerprint(PROJECT_ROOT),
            precision_mode="x64",
            jax_x64=bool(jax.config.jax_enable_x64),
            generated_artifacts=[],
            warnings=[],
            failures=[],
        )
    try:
        return _main_impl(argv)
    except Exception as exc:
        if not args.float32_debug:
            write_failed_receipt(
                receipt_path,
                phase=phase,
                command=["python", "scripts/fig06_d5_block_selection.py", *argv_list],
                argv=argv_list,
                started_utc=started_utc,
                project_root=PROJECT_ROOT,
                config_hash=conf_hash,
                exc=exc,
                failure_stage="fig06_main",
                traceback_excerpt="".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=6)),
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
