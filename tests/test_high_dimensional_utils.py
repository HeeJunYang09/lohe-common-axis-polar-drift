"""Tests for the d=5 block-selection utilities."""

from __future__ import annotations

import sys
import json
import csv
import shutil
import subprocess
import importlib.util
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from high_dimensional_utils import (
    block_weights,
    cartesian_to_polar_d5,
    common_transverse_direction,
    exact_mean_polar_drift_coefficient,
    find_persistent_fast_time,
    fit_log_weight_ratio_slope,
    integrate_d5,
    make_common_transverse_direction,
    make_d5_frequency_matrices,
    make_d5_initial_state,
    make_deterministic_two_block_samples,
    make_generic_d5_perturbations,
    make_rotation_generator,
    make_two_block_frequency_package,
    reduced_block_weights,
    reduced_hitting_time_d5,
    reduced_mean_polar_angle,
    rotate_transverse_frame,
    standardized_sample,
    validate_common_axis_structure,
)


def load_script_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_structure_and_q_block() -> None:
    package = make_two_block_frequency_package(
        16,
        sigma_std_target=(0.12, 0.27),
        bar_omega_target=(0.5, -0.25),
        mode="deterministic_independent",
    )
    J = make_rotation_generator()
    assert np.allclose(J.T, -J)
    omega = package["omega_matrices"]
    assert np.allclose(np.swapaxes(omega, 1, 2), -omega)
    assert validate_common_axis_structure(omega)["max_commutator_residual"] <= 1e-12
    assert package["Q_transverse"].shape == (4, 4)
    expected = np.diag(np.repeat(package["variance_empirical"], 2))
    assert np.linalg.norm(package["Q_transverse"] - expected, ord=2) <= 1e-12


def test_samples_and_frequency_statistics_are_empirical() -> None:
    xi, diag = make_deterministic_two_block_samples(32)
    assert np.allclose(np.mean(xi, axis=0), 0.0, atol=1e-14)
    assert np.allclose(np.mean(xi**2, axis=0), 1.0, atol=1e-14)
    assert abs(diag["correlation"]) <= 0.1
    package = make_two_block_frequency_package(
        32,
        sigma_std_target=(0.11, 0.31),
        bar_omega_target=(0.4, -0.3),
        mode="deterministic_independent",
    )
    delta = package["delta_blocks"]
    assert np.allclose(package["variance_empirical"], np.mean(delta**2, axis=0))
    assert np.allclose(package["sigma_std_empirical"], np.sqrt(package["variance_empirical"]))
    assert not np.shares_memory(package["bar_omega_target"], package["bar_omega_empirical"])
    with pytest.raises(ValueError, match="zero-variance"):
        standardized_sample(np.ones(4))


def test_initial_data_and_perturbations() -> None:
    package = make_two_block_frequency_package(
        32,
        sigma_std_target=(0.10, 0.30),
        bar_omega_target=(0.5, -0.25),
        mode="deterministic_independent",
    )
    rho = package["rho_numerator"] / 10.0
    init = make_d5_initial_state(
        package["delta_matrices"],
        10.0,
        0.85,
        (0.5, 0.5),
        (0.30, 1.05),
        rho,
        0.30,
    )
    assert np.max(np.abs(np.linalg.norm(init["u0"], axis=1) - 1.0)) <= 1e-13
    assert np.max(np.abs(np.linalg.norm(init["x0"], axis=1) - 1.0)) <= 1e-13
    assert np.all(init["x0"][:, 4] > 0.0)
    assert np.allclose(np.mean(init["zeta_phi"]), 0.0, atol=1e-14)
    assert np.allclose(np.mean(init["zeta_u"], axis=0), 0.0, atol=1e-14)
    assert np.allclose(init["zeta_u"] @ init["u_star0"], 0.0, atol=1e-14)
    scale = np.max(np.linalg.norm(init["zeta_u"], axis=1)) + np.max(np.abs(init["zeta_phi"]))
    assert np.isclose(scale, 1.0)


def test_rotating_frame_uses_empirical_means() -> None:
    ts = np.linspace(0.0, 1.0, 11)
    empirical = np.array([0.4, -0.2])
    u0 = np.array([1.0, 0.0, 0.0, 1.0])
    u = np.empty((len(ts), 1, 4))
    for n, t in enumerate(ts):
        for block, w in enumerate(empirical):
            c, s = np.cos(w * t), np.sin(w * t)
            a, b = u0[2 * block], u0[2 * block + 1]
            u[n, 0, 2 * block] = c * a - s * b
            u[n, 0, 2 * block + 1] = s * a + c * b
    recovered = rotate_transverse_frame(u, ts, empirical)
    assert np.allclose(recovered[:, 0, :], u0, atol=1e-13)
    target = empirical + np.array([0.02, 0.0])
    wrong = rotate_transverse_frame(u, ts, target)
    assert np.max(np.linalg.norm(wrong[:, 0, :] - u0, axis=1)) > 1e-3


def test_reduced_weights_slope_and_variance_usage() -> None:
    ts = np.linspace(0.0, 50.0, 501)
    variance = np.array([0.04, 0.16])
    p = reduced_block_weights(ts, 0.0, np.array([0.4, 0.6]), variance, 10.0)
    assert np.allclose(np.sum(p, axis=1), 1.0)
    assert p[-1, 0] > p[0, 0]
    equal = reduced_block_weights(ts, 0.0, np.array([0.4, 0.6]), np.array([0.09, 0.09]), 10.0)
    assert np.allclose(equal, equal[0])
    slope = fit_log_weight_ratio_slope(ts, p, 0.0, 10.0, variance, np.ones_like(ts, dtype=bool))
    assert np.isclose(slope["slope"], slope["slope_pred"], rtol=1e-3)
    wrong_variance = np.sqrt(variance)
    wrong_slope = fit_log_weight_ratio_slope(ts, p, 0.0, 10.0, wrong_variance, np.ones_like(ts, dtype=bool))
    assert abs(wrong_slope["slope"] - wrong_slope["slope_pred"]) > 0.05


def test_reduced_polar_law_and_hitting_time() -> None:
    ts = np.linspace(0.0, 80.0, 1001)
    variance = np.array([0.04, 0.16])
    p0 = np.array([0.4, 0.6])
    phi = reduced_mean_polar_angle(ts, 0.0, 0.85, p0, variance, 10.0)
    y = np.log(np.tan(phi))
    dydt = np.gradient(y, ts)
    p = reduced_block_weights(ts, 0.0, p0, variance, 10.0)
    pred = -np.sum(p * variance[None, :], axis=1) / 10.0
    assert np.allclose(dydt[10:-10], pred[10:-10], atol=5e-5)
    hit = reduced_hitting_time_d5(0.0, 0.85, 0.70, p0, variance, 10.0)
    assert hit > 0.0
    assert reduced_hitting_time_d5(0.0, 0.85, 0.85, p0, variance, 10.0) == 0.0
    with pytest.raises(ValueError):
        reduced_hitting_time_d5(0.0, 0.85, 0.90, p0, variance, 10.0)


def test_fast_time_and_exact_drift_guards() -> None:
    ts = np.linspace(0.0, 1.0, 11)
    R = np.array([1.0, 0.8, 0.2, 0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5])
    E = R.copy()
    tf, idx, reached = find_persistent_fast_time(ts, R, E, rho=0.1, K=10.0, C_tol=2.0, persistence_Kt=2.0, search_stop_index=6)
    assert reached
    assert idx == 3
    assert np.isclose(tf, ts[3])
    tf2, idx2, reached2 = find_persistent_fast_time(ts, R, E, rho=0.1, K=10.0, C_tol=2.0, persistence_Kt=2.0, search_stop_index=2)
    assert not reached2
    assert idx2 == -1

    phi = np.full((2, 2), 1e-12)
    u = np.zeros((2, 2, 4))
    u[:, :, 0] = 1.0
    out = exact_mean_polar_drift_coefficient(phi, u, 10.0, np.array([1e-12, 0.4]), eps_pole=1e-10)
    assert np.isnan(out[0])


def test_cartesian_polar_and_common_direction() -> None:
    u_star = make_common_transverse_direction((0.25, 0.75), (0.1, 0.5))
    zeta_u, zeta_phi = make_generic_d5_perturbations(16, u_star)
    assert zeta_u.shape == (16, 4)
    assert zeta_phi.shape == (16,)
    x = np.zeros((1, 16, 5))
    x[..., 0:4] = np.sin(0.8) * u_star
    x[..., 4] = np.cos(0.8)
    phi, u, sin_phi = cartesian_to_polar_d5(x)
    assert np.allclose(phi, 0.8)
    u_dir, mean_norm = common_transverse_direction(u)
    assert mean_norm[0] > 0.99
    p = block_weights(u_dir)
    assert np.allclose(p[0], [0.25, 0.75])


def test_small_integration_smoke() -> None:
    package = make_two_block_frequency_package(
        8,
        sigma_std_target=(0.05, 0.08),
        bar_omega_target=(0.2, -0.1),
        mode="deterministic_independent",
    )
    init = make_d5_initial_state(
        package["delta_matrices"],
        8.0,
        0.85,
        (0.5, 0.5),
        (0.30, 1.05),
        package["rho_numerator"] / 8.0,
        0.20,
    )
    result = integrate_d5(
        init["x0"],
        package["omega_matrices"],
        8.0,
        np.linspace(0.0, 0.1, 8),
        rtol=1e-7,
        atol=1e-9,
        dt0=1e-3,
        max_steps=5000,
    )
    assert result["x"].shape == (8, 8, 5)
    assert result["sphere_error"] < 1e-7


def test_fig06_import_is_safe_and_required_keys_unique(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pytest", "-q", "--unexpected-pytest-flag"])
    fig06 = load_script_module("fig06_import_test", "scripts/fig06_d5_block_selection.py")
    assert len(fig06.REQUIRED_CACHE_KEYS) == len(set(fig06.REQUIRED_CACHE_KEYS))
    parser = fig06._build_parser()
    args = parser.parse_args([])
    assert not args.recompute
    assert args.sigma_std_target == [0.10, 0.30]


def test_cache_validation_rejects_schema_hash_shape_and_float32() -> None:
    fig06 = load_script_module("fig06_cache_test", "scripts/fig06_d5_block_selection.py")
    cache_path = PROJECT_ROOT / "data/cache/fig06_d5_block_selection.npz"
    if not cache_path.exists():
        pytest.skip("Figure 6 cache not generated yet.")
    with np.load(cache_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    expected_hash = str(np.asarray(data["config_hash"]).item())
    fig06.validate_cache(data, expected_hash)

    wrong_schema = dict(data)
    wrong_schema["schema_version"] = np.array("bad")
    with pytest.raises(RuntimeError, match="schema_version"):
        fig06.validate_cache(wrong_schema, expected_hash)

    with pytest.raises(RuntimeError, match="config_hash"):
        fig06.validate_cache(data, "bad_hash")

    missing = dict(data)
    missing.pop("R_ans")
    with pytest.raises(RuntimeError, match="missing"):
        fig06.validate_cache(missing, expected_hash)

    wrong_shape = dict(data)
    wrong_shape["Q_transverse"] = np.zeros((5, 5))
    with pytest.raises(RuntimeError, match="Q_transverse"):
        fig06.validate_cache(wrong_shape, expected_hash)

    float32_cache = dict(data)
    float32_cache["precision_mode"] = np.array("float32_debug")
    with pytest.raises(RuntimeError, match="precision_mode"):
        fig06.validate_cache(float32_cache, expected_hash)


def test_config_hash_sensitivity() -> None:
    fig06 = load_script_module("fig06_hash_test", "scripts/fig06_d5_block_selection.py")
    args = fig06._build_parser().parse_args([])
    base = fig06.build_config(args)
    base_hash = fig06.config_hash(base)
    mutations = [
        ("theta_block_values", [0.31, 1.05]),
        ("initial_block_weights", [0.45, 0.55]),
        ("eps_pole", 1e-9),
        ("metric_phi_min", 0.36),
        ("metric_phi_max", 0.79),
        ("mean_transverse_warning", 0.91),
        ("mean_transverse_invalid", 0.49),
        ("perturbation_schema_version", "changed"),
        ("save_grid_fast_points", 802),
        ("success_thresholds", {"changed": True}),
    ]
    for key, value in mutations:
        changed = dict(base)
        changed[key] = value
        assert fig06.config_hash(changed) != base_hash


def test_fast_layer_diagnostic_table_and_scaling_helpers() -> None:
    fig06 = load_script_module("fig06_fast_diag_test", "scripts/fig06_d5_block_selection.py")
    cache_path = PROJECT_ROOT / "data/cache/fig06_d5_block_selection.npz"
    if not cache_path.exists():
        pytest.skip("Figure 6 cache not generated yet.")
    data = fig06.load_cache(cache_path, str(np.load(cache_path, allow_pickle=False)["config_hash"]))
    rows, fits = fig06.build_fast_layer_diagnostics(data)
    assert list(rows[0].keys()) == fig06.FAST_LAYER_DIAGNOSTIC_COLUMNS
    assert {row["Kt_f_Ctol_5"] for row in rows[:3]} == {0.0}
    assert all(row["fast_threshold_reached_Ctol_1"] for row in rows)
    assert 0.8 < fits["initial_R_ans"]["slope"] < 1.2
    assert 1.8 < fits["R_ans_at_Kt10"]["slope"] < 2.2


def test_output_schemas_and_failure_preservation(tmp_path) -> None:
    fig06 = load_script_module("fig06_schema_test", "scripts/fig06_d5_block_selection.py")
    assert "fast_layer_diagnostic_table.csv" not in fig06.APPENDIX_D5_CSV_SCHEMAS
    failure_path = tmp_path / "failures.json"
    payload = [{"phase": "A", "safe_to_continue": True, "message": "keep me"}]
    failure_path.write_text(json.dumps(payload), encoding="utf-8")
    assert fig06.load_existing_failures(failure_path) == payload
    missing = tmp_path / "missing.json"
    assert fig06.load_existing_failures(missing) == []


def test_appendix_status_and_completion_validation(tmp_path) -> None:
    appendix = load_script_module("appendix_status_test", "scripts/appendix_d5_diagnostics.py")
    good = {
        "log_ratio_slope_relative_error": 0.001,
        "p_relative_L2_error": 0.001,
        "drift_relative_L2_error": 0.001,
        "polar_logtan_relative_L2_error": 0.001,
        "max_phi_error": 0.001,
        "sphere_norm_error": 1e-10,
    }
    assert appendix._gaussian_status(good) == "pass"
    bad = dict(good)
    bad["polar_logtan_relative_L2_error"] = 0.07
    assert appendix._gaussian_status(bad) == "fail"

    fig06 = load_script_module("fig06_completion_test", "scripts/fig06_d5_block_selection.py")
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "figures").mkdir()
    result = fig06.validate_appendix_d5_completion(tmp_path)
    assert result["status"] == "failed"
    assert result["errors"]


def test_appendix_A1_plot_applies_paper_style(monkeypatch, tmp_path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    appendix = load_script_module("appendix_A1_style_test", "scripts/appendix_d5_diagnostics.py")
    minimal_cache = {
        "K_values": np.array([8.0, 10.0]),
        "ts": np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
        "t_f_num": np.array([0.0, 0.0]),
        "rho_values": np.array([0.05, 0.04]),
        "post_fast_valid_mask": np.ones((2, 3), dtype=bool),
        "D_u": np.array([[0.05, 0.045, 0.04], [0.04, 0.036, 0.032]]),
        "mean_transverse_norm": np.array([[0.9996, 0.9997, 0.9998], [0.9997, 0.99975, 0.9998]]),
    }
    residual_rows = [
        {"record_type": "sample", "quantity": "R_ans", "rho": 0.03, "interpolated_error": 9.0e-4},
        {"record_type": "sample", "quantity": "R_ans", "rho": 0.05, "interpolated_error": 2.5e-3},
        {"record_type": "fit", "quantity": "R_ans", "slope": 2.0, "intercept": 0.0},
        {"record_type": "sample", "quantity": "E_phi", "rho": 0.03, "interpolated_error": 8.0e-4},
        {"record_type": "sample", "quantity": "E_phi", "rho": 0.05, "interpolated_error": 2.2e-3},
        {"record_type": "fit", "quantity": "E_phi", "slope": 2.0, "intercept": 0.0},
    ]
    vector_rows = [
        {"status": "reported", "K": 8.0, "vector_L2_error": 1.0e-3, "median_alignment": 0.999999},
        {"status": "reported", "K": 10.0, "vector_L2_error": 8.0e-4, "median_alignment": 0.9999995},
    ]

    monkeypatch.setattr(appendix, "load_phase_a_cache", lambda: minimal_cache)
    monkeypatch.setattr(
        appendix,
        "save_figure_all_formats",
        lambda fig, outdir, basename: (tmp_path / f"{basename}.pdf", tmp_path / f"{basename}.png", tmp_path / f"{basename}.eps"),
    )
    mpl.rcParams.update(
        {
            "font.size": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "lines.linewidth": 0.5,
            "axes.linewidth": 0.5,
        }
    )
    appendix.plot_appendix_A1(residual_rows, vector_rows)
    try:
        assert mpl.rcParams["font.size"] == 13
        assert mpl.rcParams["axes.labelsize"] == 15
        assert mpl.rcParams["xtick.labelsize"] == 12
        assert mpl.rcParams["ytick.labelsize"] == 12
        assert mpl.rcParams["legend.fontsize"] == 11
        assert mpl.rcParams["lines.linewidth"] == 2.0
        assert mpl.rcParams["axes.linewidth"] == 1.1
    finally:
        plt.close("all")


def test_git_metadata_helper(monkeypatch) -> None:
    fig06 = load_script_module("fig06_git_test", "scripts/fig06_d5_block_selection.py")

    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(cmd, check, stdout, stderr, text):
        if "rev-parse" in cmd:
            return Completed("abc123\n")
        return Completed("")

    monkeypatch.setattr(fig06.subprocess, "run", fake_run)
    meta = fig06.get_git_metadata(PROJECT_ROOT)
    assert meta["git_commit"] == "abc123"
    assert meta["git_dirty"] is False

    def fail_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(fig06.subprocess, "run", fail_run)
    meta = fig06.get_git_metadata(PROJECT_ROOT)
    assert meta["git_commit"] == "unavailable"
    assert meta["git_dirty"] is None


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def test_phase_a_validator_with_fixture_and_corruption(tmp_path) -> None:
    fig06 = load_script_module("fig06_phase_a_validator_test", "scripts/fig06_d5_block_selection.py")
    required = [
        "data/cache/fig06_d5_block_selection.npz",
        "data/processed/metadata_exp06_d5_block_selection.json",
        "data/processed/summary_exp06_d5_block_selection.csv",
        "data/processed/d5_block_selection_summary.txt",
        "data/processed/failures_exp06_d5_block_selection.json",
        "data/processed/release_manifest_d5.txt",
        "figures/fig6_d5_block_selection.pdf",
        "figures/fig6_d5_block_selection.png",
        "figures/fig6_d5_block_selection.eps",
    ]
    if not all(_copy_if_exists(PROJECT_ROOT / rel, tmp_path / rel) for rel in required):
        pytest.skip("Phase A artifacts are not generated yet.")
    from run_receipts import CONTENT_FINGERPRINT_FILES
    for rel in CONTENT_FINGERPRINT_FILES:
        _copy_if_exists(PROJECT_ROOT / rel, tmp_path / rel)
    with np.load(tmp_path / "data/cache/fig06_d5_block_selection.npz", allow_pickle=False) as loaded:
        conf_hash = str(np.asarray(loaded["config_hash"]).item())
    from run_receipts import write_run_receipt

    generated = [tmp_path / rel for rel in required]
    write_run_receipt(
        tmp_path / "data/processed/run_receipt_phase_a_recompute.json",
        phase="phase_a_recompute",
        command=["python", "scripts/fig06_d5_block_selection.py", "--recompute", "--x64"],
        argv=["--recompute", "--x64"],
        started_utc="2026-01-01T00:00:00+00:00",
        return_code=0,
        status="passed",
        project_root=tmp_path,
        config_hash=conf_hash,
        precision_mode="x64",
        jax_x64=True,
        generated_artifacts=generated,
    )
    write_run_receipt(
        tmp_path / "data/processed/run_receipt_phase_a_cache_render.json",
        phase="phase_a_cache_render",
        command=["python", "scripts/fig06_d5_block_selection.py"],
        argv=[],
        started_utc="2026-01-01T00:00:01+00:00",
        return_code=0,
        status="passed",
        project_root=tmp_path,
        config_hash=conf_hash,
        source_fingerprint=None,
        precision_mode="x64",
        jax_x64=True,
        generated_artifacts=generated,
    )
    result = fig06.validate_phase_a_completion(tmp_path, expected_config_hash=conf_hash)
    assert result["status"] in {"passed", "completed with warnings"}

    bad_summary = tmp_path / "data/processed/summary_exp06_d5_block_selection.csv"
    bad_summary.write_text("bad\n1\n", encoding="utf-8")
    result = fig06.validate_phase_a_completion(tmp_path, expected_config_hash=conf_hash)
    assert result["status"] == "failed"
    assert any("schema" in err for err in result["errors"])


def test_appendix_d5_validator_fixture_warning_and_failed_receipt(tmp_path) -> None:
    fig06 = load_script_module("fig06_appendix_d5_validator_test", "scripts/fig06_d5_block_selection.py")
    from run_receipts import CONTENT_FINGERPRINT_FILES
    for rel in CONTENT_FINGERPRINT_FILES:
        _copy_if_exists(PROJECT_ROOT / rel, tmp_path / rel)
    required = [
        *fig06.APPENDIX_D5_REQUIRED_FILES,
        "data/processed/run_receipt_appendix_d5.json",
        "data/processed/failures_exp06_d5_block_selection.json",
        "data/processed/release_manifest_d5.txt",
        "data/cache/fig06_d5_block_selection.npz",
    ]
    if not all(_copy_if_exists(PROJECT_ROOT / rel, tmp_path / rel) for rel in required):
        pytest.skip("Appendix diagnostics artifacts are not generated yet.")
    result = fig06.validate_appendix_d5_completion(tmp_path)
    assert result["status"] in {"passed", "completed with warnings"}

    receipt = tmp_path / "data/processed/run_receipt_appendix_d5.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["return_code"] = 1
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    result = fig06.validate_appendix_d5_completion(tmp_path)
    assert result["status"] == "failed"
    assert any("return_code" in err for err in result["errors"])


def test_float32_debug_writes_no_publication_artifacts(tmp_path) -> None:
    command = [
        sys.executable,
        "scripts/fig06_d5_block_selection.py",
        "--float32-debug",
        "--cache-path",
        str(tmp_path / "cache.npz"),
        "--processed-dir",
        str(tmp_path / "processed"),
        "--output-dir",
        str(tmp_path / "figures"),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    assert not any(tmp_path.rglob("*"))


def test_source_fingerprint_changes_with_covered_file(tmp_path) -> None:
    from run_receipts import CONTENT_FINGERPRINT_FILES, compute_source_fingerprint

    for rel in CONTENT_FINGERPRINT_FILES:
        src = PROJECT_ROOT / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    base = compute_source_fingerprint(tmp_path)
    target = tmp_path / CONTENT_FINGERPRINT_FILES[0]
    target.write_text(target.read_text(encoding="utf-8") + "\n# fingerprint test\n", encoding="utf-8")
    assert compute_source_fingerprint(tmp_path) != base


def test_cache_mode_preserves_failure_json_end_to_end(tmp_path) -> None:
    cache_src = PROJECT_ROOT / "data/cache/fig06_d5_block_selection.npz"
    if not cache_src.exists():
        pytest.skip("Figure 6 cache not generated yet.")
    cache_dst = tmp_path / "data/cache/fig06_d5_block_selection.npz"
    _copy_if_exists(cache_src, cache_dst)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    failure_payload = [
        {
            "phase": "A",
            "K": 10.0,
            "frequency_mode": "deterministic_independent",
            "seed": None,
            "stage": "fixture",
            "exception_type": "FixtureWarning",
            "message": "preserve me",
            "traceback_excerpt": "",
            "safe_to_continue": True,
            "created_utc": "2026-01-01T00:00:00+00:00",
        }
    ]
    failure_path = processed / "failures_exp06_d5_block_selection.json"
    failure_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/fig06_d5_block_selection.py",
        "--cache-path",
        str(cache_dst),
        "--processed-dir",
        str(processed),
        "--output-dir",
        str(tmp_path / "figures"),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(failure_path.read_text(encoding="utf-8")) == failure_payload


def _copy_current_d5_validation_fixture(tmp_path: Path):
    fig06 = load_script_module("fig06_cv_fixture", "scripts/fig06_d5_block_selection.py")
    from run_receipts import CONTENT_FINGERPRINT_FILES

    required = set(CONTENT_FINGERPRINT_FILES)
    required.update(fig06.PHASE_A_MAIN_FILES)
    required.update(fig06.APPENDIX_D5_REQUIRED_FILES)
    required.update(
        [
            "data/processed/run_receipt_phase_a_recompute.json",
            "data/processed/run_receipt_phase_a_cache_render.json",
            "data/processed/run_receipt_appendix_d5.json",
            "data/processed/regression_fig1_4_report.json",
            "data/processed/run_receipt_regression_fig1_4.json",
            "data/processed/failures_exp06_d5_block_selection.json",
            "figures/fig6_d5_block_selection.eps",
            "paper/figure_mapping.md",
        ]
    )
    missing = [rel for rel in sorted(required) if not (PROJECT_ROOT / rel).exists()]
    if missing:
        pytest.skip(f"D5 validation fixture artifacts are not generated yet: {missing[:3]}")
    for rel in sorted(required):
        _copy_if_exists(PROJECT_ROOT / rel, tmp_path / rel)
    return fig06


def _read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_dicts(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mutate_first_csv_row(path: Path, updates: dict[str, object]) -> None:
    fieldnames, rows = _read_csv_dicts(path)
    assert rows
    rows[0].update({key: str(value) for key, value in updates.items()})
    _write_csv_dicts(path, fieldnames, rows)


def _drop_first_csv_row(path: Path) -> None:
    fieldnames, rows = _read_csv_dicts(path)
    assert rows
    _write_csv_dicts(path, fieldnames, rows[1:])


def _mutate_npz(path: Path, **updates: object) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    for key, value in updates.items():
        data[key] = np.asarray(value)
    np.savez(path, **data)


def _json_update(path: Path, updater) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    updater(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_receipt_records(root: Path, receipt_rel: str, artifact_rels: list[str] | None = None) -> None:
    receipt_path = root / receipt_rel
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    wanted = set(artifact_rels or [])
    for record in receipt.get("generated_artifacts", []):
        if artifact_rels is not None and record.get("path") not in wanted:
            continue
        artifact = root / record["path"]
        record["size_bytes"] = artifact.stat().st_size
        record["sha256"] = _sha256(artifact)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_a_result(root: Path):
    fig06 = load_script_module("fig06_cv_phase_a", "scripts/fig06_d5_block_selection.py")
    return fig06.validate_phase_a_completion(root)


def _appendix_d5_result(root: Path):
    fig06 = load_script_module("fig06_cv_appendix_d5", "scripts/fig06_d5_block_selection.py")
    return fig06.validate_appendix_d5_completion(root)


def _regression_status(root: Path) -> str:
    fig06 = load_script_module("fig06_cv_regression", "scripts/fig06_d5_block_selection.py")
    return fig06.read_recorded_regression_status(root)


def _receipt_errors(root: Path, receipt_rel: str, **kwargs) -> list[str]:
    from run_receipts import validate_receipt

    errors, _ = validate_receipt(root / receipt_rel, project_root=root, **kwargs)
    return errors


def _assert_phase_a_failed(root: Path) -> None:
    result = _phase_a_result(root)
    assert result["status"] == "failed"
    assert result["errors"] or result["warnings"]


def _assert_appendix_d5_failed(root: Path) -> None:
    result = _appendix_d5_result(root)
    assert result["status"] == "failed"
    assert result["errors"]


def _assert_regression_failed(root: Path) -> None:
    assert _regression_status(root) == "failed"


def _make_cv_test(name: str, mutate, check) -> None:
    def _test(tmp_path: Path) -> None:
        _copy_current_d5_validation_fixture(tmp_path)
        mutate(tmp_path)
        check(tmp_path)

    _test.__name__ = f"test_{name}"
    _test.__doc__ = name
    globals()[_test.__name__] = _test


def _cv01(root: Path) -> None:
    rel = "data/processed/summary_exp06_d5_block_selection.csv"
    _mutate_first_csv_row(root / rel, {"sphere_norm_error": "5e-7"})
    _refresh_receipt_records(root, "data/processed/run_receipt_phase_a_cache_render.json", [rel])


def _check_cv01(root: Path) -> None:
    result = _phase_a_result(root)
    assert result["status"] == "completed with warnings"
    assert result["scientific_status"] == "completed with warnings"


def _cv02(root: Path) -> None:
    rel = "data/processed/summary_exp06_d5_block_selection.csv"
    _mutate_first_csv_row(root / rel, {"sphere_norm_error": "2e-6"})
    _refresh_receipt_records(root, "data/processed/run_receipt_phase_a_cache_render.json", [rel])


def _check_cv02(root: Path) -> None:
    result = _phase_a_result(root)
    assert result["status"] == "failed"
    assert result["scientific_status"] == "failed"


def _cv03(root: Path) -> None:
    _mutate_npz(root / "data/cache/fig06_d5_block_selection.npz", K_values=np.array([8.0, 10.0, 12.0]))


def _cv04(root: Path) -> None:
    _mutate_npz(root / "data/cache/fig06_d5_block_selection.npz", rho_values=np.array([np.nan, 0.1, 0.1, 0.1]))


def _cv05(root: Path) -> None:
    (root / "data/processed/failures_exp06_d5_block_selection.json").write_text(
        json.dumps([{"safe_to_continue": False, "message": "unsafe"}]), encoding="utf-8"
    )


def _cv06(root: Path) -> None:
    _json_update(root / "data/processed/metadata_exp06_d5_block_selection.json", lambda payload: payload.update({"config_hash": "bad"}))


def _cv07(root: Path) -> None:
    _drop_first_csv_row(root / "data/processed/summary_exp06_d5_block_selection.csv")


def _cv08(root: Path) -> None:
    (root / "figures/fig6_d5_block_selection.pdf").unlink()


def _cv09(root: Path) -> None:
    (root / "figures/fig6_d5_block_selection.png").unlink()


def _cv10(root: Path) -> None:
    (root / "data/processed/run_receipt_phase_a_recompute.json").unlink()


def _cv11(root: Path) -> None:
    _json_update(
        root / "data/processed/run_receipt_phase_a_recompute.json",
        lambda payload: payload.update({"argv": ["--x64"], "command": ["python", "scripts/fig06_d5_block_selection.py", "--x64"]}),
    )


def _cv12(root: Path) -> None:
    _json_update(
        root / "data/processed/run_receipt_phase_a_recompute.json",
        lambda payload: payload.update({"argv": ["--recompute"], "command": ["python", "scripts/fig06_d5_block_selection.py", "--recompute"]}),
    )


def _cv13(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_phase_a_recompute.json", lambda payload: payload.update({"source_fingerprint": "bad"}))


def _cv14(root: Path) -> None:
    _mutate_npz(root / "data/cache/fig06_d5_block_selection.npz", max_axis_residual=np.array(1e-9))


def _cv15(root: Path) -> None:
    _mutate_npz(root / "data/cache/fig06_d5_block_selection.npz", config_hash=np.array("noncanonical"))
    _json_update(root / "data/processed/metadata_exp06_d5_block_selection.json", lambda payload: payload.update({"config_hash": "noncanonical"}))


def _csv_path(name: str) -> str:
    return f"data/processed/{name}"


def _cv16(root: Path) -> None:
    _mutate_first_csv_row(root / _csv_path("appendix_d5_controls.csv"), {"config_hash": "bogus"})


def _cv17(root: Path) -> None:
    _mutate_first_csv_row(root / _csv_path("appendix_d5_controls.csv"), {"config_hash": ""})


def _cv18(root: Path) -> None:
    _mutate_first_csv_row(root / _csv_path("appendix_d5_controls.csv"), {"seed": "99"})


def _cv19(root: Path) -> None:
    _mutate_first_csv_row(root / _csv_path("appendix_d5_sensitivity.csv"), {"K": "99"})


def _cv20(root: Path) -> None:
    path = root / _csv_path("appendix_d5_sensitivity.csv")
    fieldnames, rows = _read_csv_dicts(path)
    for row in rows:
        if row.get("baseline_config_hash"):
            row["refined_config_hash"] = row["baseline_config_hash"]
            break
    _write_csv_dicts(path, fieldnames, rows)


def _cv21(root: Path) -> None:
    _json_update(root / "data/processed/appendix_d5_config_registry.json", lambda payload: payload.update({"phase_a_source_config_hash": "bad"}))


def _cv22(root: Path) -> None:
    path = root / _csv_path("appendix_d5_ansatz_scaling.csv")
    fieldnames, rows = _read_csv_dicts(path)
    rows = [row for row in rows if not (row.get("record_type") == "fit" and row.get("quantity") == "R_ans")]
    _write_csv_dicts(path, fieldnames, rows)


def _cv23(root: Path) -> None:
    path = root / _csv_path("appendix_d5_sensitivity.csv")
    fieldnames, rows = _read_csv_dicts(path)
    rows = [row for row in rows if row.get("quantity") != "maximum_relative_change"]
    _write_csv_dicts(path, fieldnames, rows)


def _cv24(root: Path) -> None:
    _cv05(root)


def _cv25(root: Path) -> None:
    (root / "figures/appendix_figA2_d5_controls_robustness.pdf").unlink()


def _cv26(root: Path) -> None:
    _mutate_first_csv_row(root / _csv_path("appendix_d5_controls.csv"), {"status": "not run"})


def _cv27(root: Path) -> None:
    _drop_first_csv_row(root / _csv_path("appendix_d5_controls.csv"))


def _cv28(root: Path) -> None:
    _drop_first_csv_row(root / _csv_path("appendix_d5_sensitivity.csv"))


def _cv29(root: Path) -> None:
    _json_update(
        root / "data/processed/run_receipt_appendix_d5.json",
        lambda payload: payload.update({"argv": ["--run", "all"], "command": ["python", "scripts/appendix_d5_diagnostics.py", "--run", "all"]}),
    )


def _cv30(root: Path) -> None:
    _json_update(root / "data/processed/appendix_d5_config_registry.json", lambda payload: payload.update({"appendix_package_config_hash": "bad"}))


def _check_appendix_d5_receipt_error(root: Path) -> None:
    _assert_appendix_d5_failed(root)


def _receipt_appendix_d5_kwargs() -> dict[str, object]:
    return {
        "phase": "appendix_d5",
        "required_artifacts": ("data/processed/appendix_d5_config_registry.json",),
    }


def _cv31(root: Path) -> None:
    (root / "data/processed/appendix_d5_config_registry.json").write_text("{}\n", encoding="utf-8")


def _cv32(root: Path) -> None:
    (root / "data/processed/appendix_d5_config_registry.json").unlink()


def _cv33(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_appendix_d5.json", lambda payload: payload.update({"generated_artifacts": []}))


def _cv34(root: Path) -> None:
    def update(payload):
        payload["generated_artifacts"][0]["size_bytes"] = -1

    _json_update(root / "data/processed/run_receipt_appendix_d5.json", update)


def _cv35(root: Path) -> None:
    def update(payload):
        payload["generated_artifacts"][0]["sha256"] = "0" * 64

    _json_update(root / "data/processed/run_receipt_appendix_d5.json", update)


def _cv36(root: Path) -> None:
    def update(payload):
        payload["generated_artifacts"] = [
            record for record in payload["generated_artifacts"] if record.get("path") != "data/processed/appendix_d5_config_registry.json"
        ]

    _json_update(root / "data/processed/run_receipt_appendix_d5.json", update)


def _cv37(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_appendix_d5.json", lambda payload: payload.update({"started_utc": "not-a-timestamp"}))


def _cv38(root: Path) -> None:
    _json_update(
        root / "data/processed/run_receipt_appendix_d5.json",
        lambda payload: payload.update({"started_utc": "2026-01-02T00:00:00+00:00", "finished_utc": "2026-01-01T00:00:00+00:00"}),
    )


def _cv39(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_appendix_d5.json", lambda payload: payload.update({"finished_utc": None, "status": "passed"}))


def _check_receipt_fails(root: Path) -> None:
    errors = _receipt_errors(root, "data/processed/run_receipt_appendix_d5.json", **_receipt_appendix_d5_kwargs())
    assert errors


def _mutate_regression_report(root: Path, updater) -> None:
    _json_update(root / "data/processed/regression_fig1_4_report.json", updater)


def _cv40(root: Path) -> None:
    _mutate_regression_report(root, lambda payload: payload.update({"status": "failed"}))


def _cv41(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_regression_fig1_4.json", lambda payload: payload.update({"regression_report_sha256": "bad"}))


def _cv42(root: Path) -> None:
    def update(payload):
        payload["figure_results"][0]["status"] = "failed"

    _mutate_regression_report(root, update)


def _cv43(root: Path) -> None:
    def update(payload):
        payload["figure_results"] = payload["figure_results"][:-1]

    _mutate_regression_report(root, update)


def _cv44(root: Path) -> None:
    _mutate_regression_report(
        root,
        lambda payload: payload.update({"global_legacy_artifact_diff": {"status": "failed", "unexpected_changes": ["modified legacy artifact"]}}),
    )


def _cv45(root: Path) -> None:
    _json_update(root / "data/processed/run_receipt_regression_fig1_4.json", lambda payload: payload.update({"regression_report_path": "bad.json"}))


def _cv46(root: Path) -> None:
    _mutate_regression_report(root, lambda payload: payload.update({"mapping_status": "failed"}))


def _cv47(root: Path) -> None:
    _cv44(root)


def _cv48(root: Path) -> None:
    _mutate_regression_report(
        root,
        lambda payload: payload.update({"global_legacy_artifact_diff": {"status": "failed", "unexpected_changes": ["created legacy artifact"]}}),
    )


_CV_CASES = [
    ("cv_01_scientific_warning_produces_completed_with_warnings", _cv01, _check_cv01),
    ("cv_02_scientific_fail_is_retained_and_blocks_full_completion", _cv02, _check_cv02),
    ("cv_03_wrong_k_set_fails", _cv03, _assert_phase_a_failed),
    ("cv_04_nan_core_metric_fails", _cv04, _assert_phase_a_failed),
    ("cv_05_unsafe_failure_record_fails", _cv05, _assert_phase_a_failed),
    ("cv_06_metadata_cache_hash_mismatch_fails", _cv06, _assert_phase_a_failed),
    ("cv_07_missing_summary_row_fails", _cv07, _assert_phase_a_failed),
    ("cv_08_missing_figure6_pdf_fails", _cv08, _assert_phase_a_failed),
    ("cv_09_missing_figure6_png_fails", _cv09, _assert_phase_a_failed),
    ("cv_10_missing_recomputation_receipt_fails", _cv10, _assert_phase_a_failed),
    ("cv_11_recomputation_receipt_without_recompute_fails", _cv11, _assert_phase_a_failed),
    ("cv_12_recomputation_receipt_without_x64_fails", _cv12, _assert_phase_a_failed),
    ("cv_13_source_fingerprint_mismatch_fails", _cv13, _assert_phase_a_failed),
    ("cv_14_structural_residual_above_1e_minus_12_fails", _cv14, _assert_phase_a_failed),
    ("cv_15_internally_consistent_noncanonical_configuration_fails", _cv15, _assert_phase_a_failed),
    ("cv_16_bogus_nonempty_row_hash_fails", _cv16, _assert_appendix_d5_failed),
    ("cv_17_missing_row_hash_fails", _cv17, _assert_appendix_d5_failed),
    ("cv_18_gaussian_seed_hash_mismatch_fails", _cv18, _assert_appendix_d5_failed),
    ("cv_19_threshold_hash_mismatch_fails", _cv19, _assert_appendix_d5_failed),
    ("cv_20_tolerance_baseline_refined_hash_collision_fails", _cv20, _assert_appendix_d5_failed),
    ("cv_21_wrong_phase_a_source_hash_fails", _cv21, _assert_appendix_d5_failed),
    ("cv_22_missing_residual_fit_fails", _cv22, _assert_appendix_d5_failed),
    ("cv_23_missing_tolerance_aggregate_fails", _cv23, _assert_appendix_d5_failed),
    ("cv_24_appendix_d5_unsafe_failure_record_fails", _cv24, _assert_appendix_d5_failed),
    ("cv_25_missing_required_appendix_d5_figure_fails", _cv25, _assert_appendix_d5_failed),
    ("cv_26_not_run_row_fails", _cv26, _assert_appendix_d5_failed),
    ("cv_27_incomplete_gaussian_seed_set_fails", _cv27, _assert_appendix_d5_failed),
    ("cv_28_incomplete_threshold_grid_fails", _cv28, _assert_appendix_d5_failed),
    ("cv_29_appendix_d5_receipt_without_run_all_x64_fails", _cv29, _assert_appendix_d5_failed),
    ("cv_30_appendix_d5_package_hash_mismatch_fails", _cv30, _assert_appendix_d5_failed),
    ("cv_31_generated_artifact_modified_after_receipt_fails", _cv31, _check_receipt_fails),
    ("cv_32_generated_artifact_removed_after_receipt_fails", _cv32, _check_receipt_fails),
    ("cv_33_empty_generated_artifact_list_fails", _cv33, _check_receipt_fails),
    ("cv_34_artifact_size_mismatch_fails", _cv34, _check_receipt_fails),
    ("cv_35_artifact_sha256_mismatch_fails", _cv35, _check_receipt_fails),
    ("cv_36_required_artifact_omitted_from_receipt_fails", _cv36, _check_receipt_fails),
    ("cv_37_malformed_timestamp_fails", _cv37, _check_receipt_fails),
    ("cv_38_finish_before_start_fails", _cv38, _check_receipt_fails),
    ("cv_39_passed_receipt_without_finish_time_fails", _cv39, _check_receipt_fails),
    ("cv_40_report_failed_while_receipt_passed_fails", _cv40, _assert_regression_failed),
    ("cv_41_report_sha256_mismatch_fails", _cv41, _assert_regression_failed),
    ("cv_42_one_figure_status_failed_fails", _cv42, _assert_regression_failed),
    ("cv_43_one_required_regression_check_missing_fails", _cv43, _assert_regression_failed),
    ("cv_44_unaccepted_detected_difference_fails", _cv44, _assert_regression_failed),
    ("cv_45_receipt_report_path_mismatch_fails", _cv45, _assert_regression_failed),
    ("cv_46_missing_duplicate_or_changed_mapping_entry_fails", _cv46, _assert_regression_failed),
    ("cv_47_unexpected_legacy_scientific_modification_fails", _cv47, _assert_regression_failed),
    ("cv_48_unexpected_legacy_scientific_creation_or_deletion_fails", _cv48, _assert_regression_failed),
]

for _name, _mutate, _check in _CV_CASES:
    _make_cv_test(_name, _mutate, _check)
