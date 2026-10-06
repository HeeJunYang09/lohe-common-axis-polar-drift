"""Sensitivity of fast relaxation and slow drift to initial conditions.

Publication solvers and diagnostics are reused without modifying their source.
Default execution validates included diagnostics. Integration requires --recompute.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lohe-additional-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import diffrax
import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common_utils import (angles_to_sphere, compute_lambda_K, deterministic_frequencies,
                          first_threshold_index, fit_logtan_slope, frequency_statistics,
                          integrate_sphere_model, locking_diagnostics, logtan,
                          make_theorem_regime_initial_condition, solve_locked_profile)
import high_dimensional_utils as hd

spec = importlib.util.spec_from_file_location("initial_condition_d5", ROOT / "scripts/fig06_d5_block_selection.py")
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)
RTOL, ATOL = 1e-9, 1e-11
PAIRS = [(k, s) for k in (8., 10., 12.) for s in (.16, .20, .24)]
KS5 = (8., 10., 12., 16.)
VARIANTS = [("A", 0., -1), ("B", 0., -1)] + [
    ("C", a, seed) for a in (.05, .15, .30) for seed in range(5)]
CONFIG = {
    "schema": "initial_condition_sensitivity_v1", "rtol": RTOL, "atol": ATOL,
    "d3": {"N": 32, "pairs": PAIRS, "omega_bar": .5, "theta0": .3,
           "phi0": .85, "C_init": .30, "C_tol": 5., "T1": 220.,
           "num_save": 1800, "fit_phi_interval": [.22, .78], "fit_min_points": 8},
    "d5": {"N": 64, "K_values": KS5, "sigma": [.10, .30],
           "omega_bar": [.5, -.25], "phi0": .85, "C_init": .30, "C_tol": 5.,
           "persistence_Kt": 2., "tau_max": 18., "T_end": "1+18*K",
           "save_fast_points": 801, "save_slow_points": 2200,
           "polar_metric_interval": [.35, .80]},
    "variants": VARIANTS,
    "C_distribution": "independent uniform[-1,1] centered and max-normalized polar/3D azimuth; paired Gaussian tangent directions in 5D",
    "C_5D_tangent": "h,-h pairs with random radii, random particle permutation, max norm=1; u_i=normalize(u_star+a*h_i)",
    "reference": "data/cache",
    "unreached_policy": "no minimum-error fallback; unresolved post-fast metrics remain NaN",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def signature():
    files = [ROOT / "src/common_utils.py", ROOT / "src/high_dimensional_utils.py",
             ROOT / "scripts/fig06_d5_block_selection.py", Path(__file__)]
    payload = {"config": CONFIG, "sources": {str(p.relative_to(ROOT)): sha(p) for p in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def centered_random(rng, n):
    x = rng.uniform(-1., 1., n)
    x -= x.mean()
    return x / np.max(np.abs(x))


def init3(omega, k, profile, group, amplitude, seed):
    if group in ("A", "B"):
        v = profile if group == "A" else np.zeros_like(profile)
        return make_theorem_regime_initial_condition(omega, k, v, .3, .85, .30)[0]
    rng = np.random.default_rng(seed)
    theta = .3 + amplitude * centered_random(rng, 32)
    phi = .85 + amplitude * centered_random(rng, 32)
    return angles_to_sphere(theta, phi)


def init5(package, k, group, amplitude, seed):
    rho = package["rho_numerator"] / k
    base = hd.make_d5_initial_state(package["delta_matrices"], k, .85,
                                    (.5, .5), (.3, 1.05), rho, .30)
    if group == "A":
        return base
    ustar = base["u_star0"]
    if group == "B":
        v = ustar[None, :] + .30 * rho * base["zeta_u"]
        phi = base["phi0_i"]
    else:
        rng = np.random.default_rng(seed)
        h = rng.normal(size=(32, 4))
        h -= (h @ ustar)[:, None] * ustar
        h /= np.linalg.norm(h, axis=1)[:, None]
        h *= rng.uniform(.1, 1., 32)[:, None]
        h /= np.max(np.linalg.norm(h, axis=1))
        h = np.concatenate([h, -h])[rng.permutation(64)]
        v = ustar[None, :] + amplitude * h
        phi = .85 + amplitude * centered_random(rng, 64)
    u = v / np.linalg.norm(v, axis=1)[:, None]
    x = np.column_stack([np.sin(phi)[:, None] * u, np.cos(phi)])
    return {**base, "x0": x, "u0": u, "phi0_i": phi}


def self_check():
    package = hd.make_two_block_frequency_package(64, (.1, .3), (.5, -.25), "deterministic_independent")
    checks = 0
    for group, amp, seed in VARIANTS:
        for k in KS5:
            initial = init5(package, k, group, amp, seed)
            x = initial["x0"]
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1., atol=1e-14, rtol=0)
            np.testing.assert_allclose(initial["phi0_i"].mean(), .85, atol=1e-14, rtol=0)
            assert np.all(x[:, -1] > 0)
            if group == "C":
                actual = initial["u0"].mean(axis=0)
                actual /= np.linalg.norm(actual)
                np.testing.assert_allclose(actual, initial["u_star0"], atol=1e-14, rtol=0)
            if group == "B":
                base = init5(package, k, "A", 0., -1)
                np.testing.assert_array_equal(initial["phi0_i"], base["phi0_i"])
            checks += 1
        for k, sigma in PAIRS:
            omega = deterministic_frequencies(32, .5, sigma)
            profile, _ = solve_locked_profile(omega, k)
            x = init3(omega, k, profile, group, amp, seed)
            diag = locking_diagnostics(np.array([0.]), x[None, :, :], .5, profile)
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1., atol=1e-14, rtol=0)
            np.testing.assert_allclose(diag["phi_bar"][0], .85, atol=1e-14, rtol=0)
            np.testing.assert_allclose(diag["theta"][0].mean(), .3, atol=1e-14, rtol=0)
            assert np.all(x[:, -1] > 0)
            if group == "B":
                base = init3(omega, k, profile, "A", 0., -1)
                d0 = locking_diagnostics(np.array([0.]), base[None], .5, profile)
                np.testing.assert_allclose(diag["theta"][0] + profile, d0["theta"][0], atol=1e-14, rtol=0)
                np.testing.assert_allclose(diag["phi"][0], d0["phi"][0], atol=1e-14, rtol=0)
            checks += 1
    result = {"passed": True, "initializations_checked": checks,
              "jax_x64": bool(jax.config.jax_enable_x64), "jax": jax.__version__,
              "diffrax": diffrax.__version__, "numpy": np.__version__}
    write_json(OUT / "_work/self_check.json", result)
    print("PASS self-check", result, flush=True)


def run3(k, sigma, group, amp, seed):
    omega = deterministic_frequencies(32, .5, sigma)
    stats = frequency_statistics(omega, k)
    profile, residual = solve_locked_profile(omega, k)
    coefficient = compute_lambda_K(profile, k)
    x0 = init3(omega, k, profile, group, amp, seed)
    sol = integrate_sphere_model(omega, k, x0, 0., 220., 1800, rtol=RTOL, atol=ATOL)
    diag = locking_diagnostics(sol.ts, sol.xs, stats["omega_bar"], profile)
    threshold = 5. * stats["rho"] ** 2
    idx = first_threshold_index(diag["E_lock"], threshold)
    reached = idx is not None
    tf = float(sol.ts[idx]) if reached else float("nan")
    slope, intercept, mask = fit_logtan_slope(sol.ts, diag["phi_bar"], tf, .22, .78, min_points=8)
    Y = logtan(diag["phi_bar"])
    fit_valid = bool(reached and np.isfinite(slope) and mask.sum() >= 8)
    ss = float(np.sum((Y[mask] - Y[mask].mean())**2)) if fit_valid else float("nan")
    r2 = 1. - np.sum((Y[mask] - slope * sol.ts[mask] - intercept)**2) / ss if fit_valid and ss > 0 else float("nan")
    row = {"dimension": 3, "N": 32, "K": k, "sigma_omega": sigma,
           "rho": stats["rho"], "Lambda_K": coefficient, "locked_profile_residual": residual,
           "t_f_num": tf, "K_t_f_num": k * tf, "threshold": threshold,
           "threshold_reached": reached, "threshold_maintained": bool(reached and np.all(diag["E_lock"][idx:] <= threshold)),
           "initial_error": float(diag["E_lock"][0]), "final_error": float(diag["E_lock"][-1]),
           "initial_phi_bar": float(diag["phi_bar"][0]),
           "phi_bar_tf": float(diag["phi_bar"][idx]) if reached else float("nan"),
           "initial_phi_min": float(diag["phi"][0].min()), "initial_phi_max": float(diag["phi"][0].max()),
           "initial_theta_mean": float(diag["theta"][0].mean()),
           "initial_theta_spread": float(np.max(np.abs(diag["theta"][0]-.3))),
           "initial_polar_spread": float(diag["E_phi"][0]),
           "fit_slope": float(slope), "fit_intercept": float(intercept), "fit_r_squared": float(r2),
           "fit_count": int(mask.sum()), "fit_phi_min": .22, "fit_phi_max": .78,
           "D_meas": -float(slope), "rate_ratio": -float(slope)/coefficient,
           "relative_rate_error": abs(-float(slope)/coefficient - 1.),
           "metric_valid": fit_valid, "sphere_error_max": sol.stats["sphere_norm_error"],
           "state_dtype": str(sol.xs.dtype), "time_dtype": str(sol.ts.dtype)}
    arrays = {"ts": sol.ts, "states": sol.xs, "initial_state": x0, "omega": omega,
              "locked_profile": profile, "phi_bar": diag["phi_bar"], "E_lock": diag["E_lock"],
              "E_theta": diag["E_theta"], "E_phi": diag["E_phi"], "Y": Y, "fit_mask": mask}
    return row, arrays


def run5(k, group, amp, seed):
    package = hd.make_two_block_frequency_package(64, (.1, .3), (.5, -.25), "deterministic_independent")
    initial = init5(package, k, group, amp, seed)
    captured = {}
    original_init, original_integrate = d5.make_d5_initial_state, d5.integrate_d5

    def capture(*args, **kwargs):
        result = original_integrate(*args, **kwargs)
        captured["states"] = result["x"]
        return result

    # Override only the initial state; retain the exact publication run_case diagnostics.
    try:
        d5.make_d5_initial_state = lambda *args, **kwargs: initial
        d5.integrate_d5 = capture
        result = d5.run_case(d5._build_parser().parse_args([]), k, package)
    finally:
        d5.make_d5_initial_state, d5.integrate_d5 = original_init, original_integrate
    reached = result["fast_threshold_reached"]
    idx, tf = result["t_f_index"], result["t_f_num"]
    row = dict(result["summary_row"])
    threshold = 5. * result["rho"]**2
    slope, predicted = row["log_ratio_slope_fit"], row["log_ratio_slope_pred"]
    mean0 = initial["u0"].mean(axis=0)
    mean0 /= np.linalg.norm(mean0)
    row.update({"dimension": 5, "N": 64, "threshold": threshold, "threshold_reached": reached,
                "threshold_maintained": bool(reached and np.all(result["R_ans"][idx:] <= threshold)
                                             and np.all(result["E_phi"][idx:] <= threshold)),
                "initial_error": float(max(result["R_ans"][0], result["E_phi"][0])),
                "initial_R_ans": float(result["R_ans"][0]), "initial_E_phi": float(result["E_phi"][0]),
                "final_error": float(max(result["R_ans"][-1], result["E_phi"][-1])),
                "initial_phi_bar": float(result["phi_bar"][0]),
                "phi_bar_tf": float(result["phi_bar"][idx]) if reached else float("nan"),
                "initial_phi_min": float(initial["phi0_i"].min()), "initial_phi_max": float(initial["phi0_i"].max()),
                "initial_p1": float(result["p_sim"][0, 0]), "initial_p2": float(result["p_sim"][0, 1]),
                "initial_mean_direction_error": float(np.linalg.norm(mean0 - initial["u_star0"])),
                "fit_slope": slope, "fit_r_squared": row["log_ratio_r_squared"],
                "fit_count": int(np.sum(result["post_fast_valid_mask"])),
                "rate_ratio": slope / predicted, "relative_rate_error": abs(slope / predicted - 1.),
                "metric_valid": bool(reached and np.isfinite(slope) and row["polar_metric_point_count"] >= 100),
                "sphere_error_max": result["sphere_norm_error"],
                "state_dtype": str(captured["states"].dtype), "time_dtype": str(result["ts"].dtype)})
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(captured)
    arrays.update({"omega_matrices": package["omega_matrices"], "variance_empirical": package["variance_empirical"]})
    return row, arrays


def reference_check(row, arrays):
    checks = {}
    dimension = row["dimension"]
    name = "fig03_slow_polar_drift_deterministic_3x3.npz" if dimension == 3 else "fig06_d5_block_selection.npz"
    path = REFERENCE / "data/cache" / name
    with np.load(path, allow_pickle=False) as z:
        if dimension == 3:
            i = int(np.flatnonzero((z["panel_b_K"] == row["K"]) & (z["panel_b_sigma"] == row["sigma_omega"]))[0])
            pairs = {"D_meas": "panel_b_D_meas", "Lambda_K": "panel_b_Lambda_K",
                     "rate_ratio": "panel_b_D_meas_over_Lambda_K", "t_f_num": "panel_b_tf_num",
                     "sphere_error_max": "panel_b_sphere_norm_error", "rho": "panel_b_rho"}
            values = [(key, row[key], z[value][i]) for key, value in pairs.items()]
            values += [("ts", arrays["ts"], z["ts"])]
        else:
            i = int(np.flatnonzero(z["K_values"] == row["K"])[0])
            names = ["ts", "phi_bar", "phi_red", "E_phi", "R_ans", "p_sim", "p_red", "post_fast_valid_mask", "polar_metric_valid_mask"]
            values = [(key, arrays[key], z[key][i]) for key in names]
            values += [("initial_state", arrays["initial_state"], z["initial_states"][i]),
                       ("t_f_num", row["t_f_num"], z["t_f_num"][i])]
        for key, actual, expected in values:
            a, b = np.asarray(actual), np.asarray(expected)
            checks[key] = {"passed": bool(np.allclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True)),
                           "exact_equal": bool(np.array_equal(a, b, equal_nan=True))}
    return {"passed": all(c["passed"] for c in checks.values()), "reference_file": str(path.relative_to(ROOT)),
            "reference_sha256": sha(path), "rtol": 1e-9, "atol": 1e-11, "fields": checks}


def cases():
    result = []
    for group, amp, seed in VARIANTS:
        for dim, conditions in ((3, PAIRS), (5, [(k, 0.) for k in KS5])):
            for k, sigma in conditions:
                key = f"d{dim}_K{k:g}_s{sigma:.2f}_{group}_a{amp:.2f}_seed{seed}"
                result.append((key, dim, k, sigma, group, amp, seed))
    return result



def load_diagnostics(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "row_json"}
        row = json.loads(str(z["row_json"].item()))
    return row, arrays


def export_case(path, row, arrays):
    # Retain all saved diagnostics, but not the large full-state trajectories.
    compact = {k: v for k, v in arrays.items()
               if isinstance(v, np.ndarray) and k not in {"states", "signature", "row_json", "regression_json"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **compact, row_json=np.array(json.dumps(clean(row))))


def validate():
    rows, regressions = [], []
    frequencies, time_grids, random_states = {}, {}, {}
    for key, dim, k, sigma, group, amp, seed in cases():
        path = OUT / "data/diagnostics" / (key + ".npz")
        row, arrays = load_diagnostics(path)
        assert row["case_id"] == key and row["dimension"] == dim and row["group"] == group
        assert row["K"] == k and row["initial_seed"] == seed
        assert row["amplitude"] == (amp if group == "C" else None)
        if dim == 3:
            assert row["sigma_omega"] == sigma
        assert row["threshold_reached"] and row["threshold_maintained"] and row["metric_valid"]
        ts = arrays["ts"]
        assert ts.shape == ((1800,) if dim == 3 else (3000,)) and ts.dtype == np.float64
        assert np.all(np.isfinite(ts)) and np.all(np.diff(ts) > 0)
        initial = arrays["initial_state"]
        np.testing.assert_allclose(np.linalg.norm(initial, axis=1), 1., rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.arccos(initial[:, -1]).mean(), .85, rtol=0, atol=1e-14)
        assert np.all(initial[:, -1] > 0)
        condition = (dim, k, sigma)
        omega = arrays["omega"] if dim == 3 else arrays["omega_matrices"]
        for mapping, value in ((frequencies, omega), (time_grids, ts)):
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            assert mapping.setdefault(condition, digest) == digest
        if group == "C":
            digest = hashlib.sha256(initial.tobytes()).hexdigest()
            assert random_states.setdefault((dim, amp, seed), digest) == digest
        if dim == 3:
            slope, _, mask = fit_logtan_slope(ts, arrays["phi_bar"], row["t_f_num"], .22, .78, min_points=8)
            np.testing.assert_array_equal(mask, arrays["fit_mask"])
            np.testing.assert_allclose(-slope, row["D_meas"], rtol=1e-12, atol=1e-14)
        else:
            fit = hd.fit_log_weight_ratio_slope(ts, arrays["p_sim"], row["t_f_num"], k,
                                              arrays["variance_empirical"], arrays["post_fast_valid_mask"])
            slope = fit["slope"]
        np.testing.assert_allclose(slope, row["fit_slope"], rtol=1e-12, atol=1e-14)
        error = arrays["E_lock"] if dim == 3 else np.maximum(arrays["R_ans"], arrays["E_phi"])
        idx = int(np.searchsorted(ts, row["t_f_num"]))
        np.testing.assert_allclose(ts[idx], row["t_f_num"], rtol=0, atol=1e-14)
        assert np.all(error[idx:] <= row["threshold"])
        np.testing.assert_allclose(row["relative_rate_error"], abs(row["rate_ratio"]-1), rtol=1e-13)
        if group == "A":
            check = reference_check(row, arrays)
            assert check["passed"], key
            regressions.append({"case_id": key, **check})
        rows.append(row)
    assert len(rows) == 221 and len(regressions) == 13
    print("PASS 221 compact diagnostics, initial states, fixed frequencies/grids, refitted slopes, "
          "maintained thresholds, and 13 manuscript baselines.")
    return rows, regressions


def collect():
    rows, regressions = [], []
    for key, *_ in cases():
        path = OUT / "data/diagnostics" / (key + ".npz")
        if not path.exists():
            continue
        row, arrays = load_diagnostics(path)
        rows.append(row)
        if row["group"] == "A":
            regressions.append({"case_id": key, **reference_check(row, arrays)})
    for dim in (3, 5):
        subset = [r for r in rows if r["dimension"] == dim]
        if not subset:
            continue
        columns = sorted(set().union(*(r.keys() for r in subset)))
        with (OUT / f"data/d{dim}_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(subset)
    write_json(OUT / "data/baseline_regression.json",
               {"expected": 13, "completed": len(regressions),
                "passed": len(regressions) == 13 and all(r["passed"] for r in regressions),
                "cases": regressions})
    write_json(OUT / "data/metadata.json",
               {"configuration": CONFIG, "expected_cases": 221, "completed_cases": len(rows),
                "cases": rows, "python": sys.version, "jax": jax.__version__,
                "diffrax": diffrax.__version__, "numpy": np.__version__,
                "jax_x64": bool(jax.config.jax_enable_x64),
                "storage": "Compact saved diagnostics; full particle states are local _work artifacts."})
    return rows


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Integrate selected cases; resume matching local _work caches.")
    parser.add_argument("--stage", choices=["self-check", "baselines", "sanity", "full"], default="full")
    parser.add_argument("--output-dir", type=Path, help="Separate experiment output directory.")
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir.resolve()
    if args.stage == "self-check":
        self_check()
        return
    if not args.recompute:
        validate()
        return
    for sub in ("data/diagnostics", "_work/cache", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    selected = cases()
    if args.stage == "baselines":
        selected = [c for c in selected if c[4] == "A"]
    elif args.stage == "sanity":
        selected = [c for c in selected if (c[1:4] == (3, 10., .20) or c[1] == 5 and c[2] == 12.)
                    and (c[4] in ("A", "B") or c[5:] == (.30, 0))]
    if args.stage == "full":
        selected.sort(key=lambda c: c[4] != "A")
    for index, (key, dim, k, sigma, group, amp, seed) in enumerate(selected):
        path = OUT / "_work/cache" / (key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["signature"].item()) != signature():
                    raise RuntimeError(f"Stale local cache: {path}. Use a fresh --output-dir.")
                row = json.loads(str(z["row_json"].item()))
                arrays = {n: z[n] for n in z.files if n not in {"signature", "row_json"}}
        else:
            start = time.monotonic()
            print(f"START {index+1}/{len(selected)} {key}", flush=True)
            row, arrays = run3(k, sigma, group, amp, seed) if dim == 3 else run5(k, group, amp, seed)
            row.update({"case_id": key, "group": group, "amplitude": amp if group == "C" else None,
                        "initial_seed": seed, "runtime_seconds": time.monotonic()-start})
            np.savez_compressed(path, **arrays, signature=np.array(signature()),
                                row_json=np.array(json.dumps(clean(row))))
        if group == "A" and not reference_check(row, arrays)["passed"]:
            raise RuntimeError(f"Publication regression failed for {key}")
        export_case(OUT / "data/diagnostics" / (key + ".npz"), row, arrays)
        print(f"DONE {key}", flush=True)
    rows = collect()
    if len(rows) == 221:
        validate()
    print(f"Completed {len(rows)}/221 cases.", flush=True)


if __name__ == "__main__":
    main()
