"""Multi-target hitting-time validation with the manuscript numerical settings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lohe-additional-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import diffrax
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from common_utils import (
    compute_lambda_K,
    deterministic_frequencies,
    first_threshold_index,
    frequency_statistics,
    integrate_sphere_model,
    interpolated_hitting_time,
    locking_diagnostics,
    logtan,
    make_theorem_regime_initial_condition,
    save_metadata,
    solve_locked_profile,
)
from figure_style import apply_paper_style, format_axes

OUTPUT = Path(__file__).resolve().parents[1]
PUBLICATION_ROOT = ROOT
REFERENCE_NAME = "fig04_hitting_time_deterministic_5x5.npz"
N = 28
K_VALUES = np.array([6.0, 7.5, 9.0, 10.5, 12.0])
SIGMA_VALUES = np.array([0.20, 0.24, 0.28, 0.32, 0.36])
ETAS = np.array([0.84, 0.82, 0.70, 0.50, 0.30])
RTOL, ATOL = 1e-9, 1e-11
CONFIG = {
    "schema": "multi_target_hitting_time_x64_v1",
    "N": N,
    "K_values": K_VALUES.tolist(),
    "sigma_values": SIGMA_VALUES.tolist(),
    "eta_values": ETAS.tolist(),
    "omega_bar": 0.5,
    "theta0": 0.3,
    "phi0": 0.85,
    "c_init": 0.30,
    "C_tol": 5.0,
    "rtol": RTOL,
    "atol": ATOL,
    "baseline_horizon": 300.0,
    "baseline_num_save": 2200,
    "continuation_horizon": 450.0,
    "continuation_num_save": 1101,
}
PAIRS = [(float(k), float(s)) for k in K_VALUES for s in SIGMA_VALUES]
REFERENCE_TOLERANCES = {
    "tf_num": 1e-12,
    "phi_tf": 1e-10,
    "Lambda_K": 1e-14,
    "T_sim": 1e-8,
    "T_pred": 1e-8,
    "relative_error": 1e-8,
}
ERROR_DEFINITIONS = {
    "relative_error_total": "abs(T_sim-T_pred)/T_sim",
    "relative_error_slow": "abs((T_sim-t_f)-(T_pred-t_f))/(T_pred-t_f)",
    "relative_error_interval": "abs(interval_sim-interval_pred)/interval_pred",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_data():
    path = PUBLICATION_ROOT / "data/cache" / REFERENCE_NAME
    with np.load(path, allow_pickle=False) as loaded:
        data = dict(loaded)
    if str(data.get("precision_mode", "")) != "x64":
        raise ValueError("The publication reference must be the final x64 cache.")
    if not np.array_equal(data["K_values"], K_VALUES) or not np.array_equal(
        data["sigma_values"], SIGMA_VALUES
    ):
        raise ValueError("Publication parameter grid does not match this experiment.")
    return path, data


def cache_path(index, refined=False):
    suffix = "_refined" if refined else ""
    return OUTPUT / "_work" / "cache" / f"case_{index:02d}{suffix}.npz"


def simulate(index, refined=False):
    started = time.monotonic()
    k, sigma = PAIRS[index]
    omega = deterministic_frequencies(N, omega_bar=0.5, sigma_omega=sigma)
    stats = frequency_statistics(omega, k)
    vartheta, locked_residual = solve_locked_profile(omega, k)
    coefficient = compute_lambda_K(vartheta, k)
    x0, _ = make_theorem_regime_initial_condition(
        omega, k, vartheta, theta0=0.3, phi0=0.85, perturb_scale=0.30
    )
    first_count = 4399 if refined else 2200
    second_count = 2201 if refined else 1101
    # Use the unmodified publication integration call before continuing at t=300.
    first = integrate_sphere_model(
        omega, k, x0, t0=0.0, t1=300.0, num_save=first_count, rtol=RTOL, atol=ATOL
    )
    second = integrate_sphere_model(
        omega, k, first.xs[-1], t0=300.0, t1=450.0,
        num_save=second_count, rtol=RTOL, atol=ATOL
    )
    ts = np.concatenate((first.ts, second.ts[1:]))
    xs = np.concatenate((first.xs, second.xs[1:]), axis=0)
    diag = locking_diagnostics(ts, xs, stats["omega_bar"], vartheta)
    threshold = 5.0 * stats["rho"] ** 2
    if refined:
        with np.load(cache_path(index), allow_pickle=False) as coarse:
            tf_num = float(coarse["tf_num"])
        phi_tf = float(np.interp(tf_num, ts, diag["phi_bar"]))
        reached = True
    else:
        idx = first_threshold_index(diag["E_lock"][:2200], threshold)
        reached = idx is not None
        if not reached:
            raise RuntimeError(f"Fast threshold not reached for K={k}, sigma={sigma}.")
        tf_num = float(ts[idx])
        phi_tf = float(diag["phi_bar"][idx])
    if phi_tf <= float(ETAS.max()):
        raise RuntimeError("A target is already reached at the fast-time anchor.")
    payload = {
        "configuration": np.array(json.dumps(CONFIG, sort_keys=True)),
        "refined": np.array(refined), "index": np.array(index),
        "K": np.array(k), "sigma_omega": np.array(sigma),
        "rho": np.array(stats["rho"]), "omega": omega, "vartheta": vartheta,
        "Lambda_K": np.array(coefficient), "locked_profile_residual": np.array(locked_residual),
        "ts": ts, "xs": xs, "phi_bar": diag["phi_bar"],
        "E_lock": diag["E_lock"], "E_theta": diag["E_theta"], "E_phi": diag["E_phi"],
        "phi_min": np.min(diag["phi"], axis=1), "phi_max": np.max(diag["phi"], axis=1),
        "tf_num": np.array(tf_num), "phi_tf": np.array(phi_tf),
        "fast_threshold": np.array(threshold), "fast_threshold_reached": np.array(reached),
        "sphere_norm_error_max": np.array(max(first.stats["sphere_norm_error"], second.stats["sphere_norm_error"])),
        "prefix_sphere_norm_error": np.array(first.stats["sphere_norm_error"]),
        "runtime_seconds": np.array(time.monotonic() - started),
    }
    np.savez_compressed(cache_path(index, refined), **payload)
    return payload


def load_or_simulate(index, refined=False):
    path = cache_path(index, refined)
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            data = dict(loaded)
        if str(data["configuration"]) != json.dumps(CONFIG, sort_keys=True):
            raise ValueError(f"Cache configuration mismatch: {path}")
        if int(data["index"]) != index or bool(data["refined"]) != refined:
            raise ValueError(f"Cache case mismatch: {path}")
        return data, True
    return simulate(index, refined), False


def target_records(data):
    ts, phi = data["ts"], data["phi_bar"]
    tf, phi_tf = float(data["tf_num"]), float(data["phi_tf"])
    coefficient = float(data["Lambda_K"])
    reference_time = interpolated_hitting_time(ts, phi, 0.82, start_time=tf)
    rows = []
    for eta in ETAS:
        measured = interpolated_hitting_time(ts, phi, float(eta), start_time=tf)
        logarithm = float(np.log(np.tan(phi_tf) / np.tan(eta)))
        prediction = tf + logarithm / coefficient
        reached = bool(np.isfinite(measured))
        slow_time = measured - tf
        scaled_time = coefficient * slow_time
        interval_log = float(logtan(0.82) - logtan(eta)) if eta < 0.82 else float("nan")
        interval_time = measured - reference_time if eta < 0.82 else float("nan")
        window = (ts >= tf) & (ts <= measured) if reached else np.zeros(ts.shape, dtype=bool)
        norm = np.max(np.abs(np.linalg.norm(data["xs"][window], axis=-1) - 1.0)) if window.any() else float("nan")
        rows.append({
            "case_index": int(data["index"]), "N": N,
            "K": float(data["K"]), "sigma_omega": float(data["sigma_omega"]),
            "rho": float(data["rho"]), "Lambda_K": coefficient, "eta": float(eta),
            "t_f_num": tf, "phi_tf": phi_tf,
            "fast_threshold": float(data["fast_threshold"]),
            "fast_threshold_reached": bool(data["fast_threshold_reached"]),
            "threshold_reached": reached, "T_eta_sim": measured, "T_eta_pred": prediction,
            "slow_time_sim": slow_time, "slow_time_pred": logarithm / coefficient,
            "log_factor": logarithm, "scaled_slow_time": scaled_time,
            "relative_error_total": abs(measured - prediction) / measured if reached else float("nan"),
            "relative_error_slow": abs(scaled_time - logarithm) / logarithm if reached else float("nan"),
            "signed_relative_error_slow": (scaled_time - logarithm) / logarithm if reached else float("nan"),
            "interval_time_from_eta082": interval_time,
            "interval_log_factor": interval_log,
            "scaled_interval_time": coefficient * interval_time,
            "relative_error_interval": abs(coefficient * interval_time - interval_log) / interval_log
            if eta < 0.82 and reached and np.isfinite(reference_time) else float("nan"),
            "max_E_lock_over_rho_squared": float(np.max(data["E_lock"][window]) / float(data["rho"]) ** 2)
            if window.any() else float("nan"),
            "locking_threshold_maintained_at_saved_times": bool(np.all(data["E_lock"][window] <= float(data["fast_threshold"])))
            if window.any() else False,
            "individual_phi_min_before_hit": float(data["phi_min"][window].min()) if window.any() else float("nan"),
            "individual_phi_max_before_hit": float(data["phi_max"][window].max()) if window.any() else float("nan"),
            "sphere_norm_error_before_hit": float(norm),
            "sphere_norm_error_max": float(data["sphere_norm_error_max"]),
            "locked_profile_residual": float(data["locked_profile_residual"]),
        })
    return rows


def regression(data, reference):
    row = next(r for r in target_records(data) if r["eta"] == 0.82)
    index = int(data["index"])
    if reference["K_case"][index] != row["K"] or reference["sigma_case"][index] != row["sigma_omega"]:
        raise ValueError("Reference case ordering mismatch.")
    fields = {"tf_num": row["t_f_num"], "phi_tf": row["phi_tf"], "Lambda_K": row["Lambda_K"],
              "T_sim": row["T_eta_sim"], "T_pred": row["T_eta_pred"],
              "relative_error": row["relative_error_total"]}
    checks = {}
    for field, value in fields.items():
        expected = float(reference[field][index])
        checks[field] = {
            "actual": value, "reference": expected, "absolute_difference": abs(value - expected),
            "exact_equal": value == expected, "atol": REFERENCE_TOLERANCES[field], "rtol": 1e-9,
            "passed": bool(np.isclose(value, expected, atol=REFERENCE_TOLERANCES[field], rtol=1e-9)),
        }
    result = {"case_index": index, "K": row["K"], "sigma_omega": row["sigma_omega"],
              "fields": checks, "passed": all(c["passed"] for c in checks.values()),
              "all_fields_exact_equal": all(c["exact_equal"] for c in checks.values())}
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    output = []
    for eta in ETAS:
        selected = [r for r in rows if r["eta"] == eta]
        valid = [r for r in selected if r["threshold_reached"]]
        record = {"eta": float(eta), "case_count": len(selected), "reached_count": len(valid)}
        for field in ("relative_error_total", "relative_error_slow", "relative_error_interval"):
            values = np.array([r[field] for r in valid])
            values = values[np.isfinite(values)]
            for metric, function in (("min", np.min), ("mean", np.mean), ("max", np.max)):
                record[f"{field}_{metric}"] = float(function(values)) if values.size else float("nan")
        output.append(record)
    return output


def plot(rows):
    apply_paper_style()
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})
    colors = ["#CC79A7", "#0072B2", "#D55E00", "#009E73", "#000000"]
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.97, bottom=0.22, wspace=0.35)
    x = np.linspace(0, max(r["log_factor"] for r in rows) * 1.045, 200)
    for ax in axes:
        ax.plot(x, x, "k--", lw=2.2, zorder=5, label="Leading-order prediction")
        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, x[-1])
        ax.set_box_aspect(1)
        format_axes(ax)
    for eta, color, marker in zip(ETAS, colors, markers):
        subset = [r for r in rows if r["eta"] == eta and r["threshold_reached"]]
        axes[0].scatter([r["log_factor"] for r in subset], [r["scaled_slow_time"] for r in subset],
                        s=42, marker=marker, color=color, facecolors="none", linewidths=1.2,
                        zorder=7, label=rf"$\eta={eta:.2f}$")
        if eta < 0.82:
            axes[1].scatter([r["interval_log_factor"] for r in subset],
                            [r["scaled_interval_time"] for r in subset], s=42, marker=marker,
                            color=color, facecolors="none", linewidths=1.2, zorder=7,
                            label=rf"$\eta={eta:.2f}$")
    axes[0].set_xlabel(r"$\log[\tan\bar\phi(t_f^{\mathrm{num}})/\tan\eta]$")
    axes[0].set_ylabel(r"$\Lambda_K(T_\eta^{\mathrm{sim}}-t_f^{\mathrm{num}})$")
    axes[1].set_xlabel(r"$\log[\tan(0.82)/\tan\eta]$")
    axes[1].set_ylabel(r"$\Lambda_K(T_\eta^{\mathrm{sim}}-T_{0.82}^{\mathrm{sim}})$")
    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.text(0.5, -0.25, label, transform=ax.transAxes, ha="center", va="top", fontsize=15)
        ax.legend(loc="upper left", handlelength=1.6)
    save_plot(fig, "fig_logarithmic_law")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.96, bottom=0.22, wspace=0.34)
    representatives = [(12.0, 0.20), (9.0, 0.28), (6.0, 0.36)]
    for (k, sigma), color, marker in zip(representatives, colors[1:4], markers):
        subset = sorted([r for r in rows if r["K"] == k and r["sigma_omega"] == sigma], key=lambda r: r["eta"])
        coefficient, tf, phi_tf = subset[0]["Lambda_K"], subset[0]["t_f_num"], subset[0]["phi_tf"]
        target = np.linspace(0.30, 0.84, 200)
        prediction = tf + (logtan(phi_tf) - logtan(target)) / coefficient
        axes[0].plot(target, prediction, color=color, linestyle="-", lw=1.8)
        axes[0].scatter([r["eta"] for r in subset], [r["T_eta_sim"] for r in subset],
                        color=color, marker=marker, s=45, zorder=5,
                        label=rf"$K={k:g},\ \sigma_\omega={sigma:.2f}$")
    axes[0].set(xlabel=r"Target polar angle $\eta$", ylabel=r"$T_\eta^{\mathrm{sim}}$",
                xlim=(0.28, 0.86), ylim=(0, 420))
    axes[0].legend(loc="upper right")
    ascending = np.sort(ETAS)
    for metric, color, marker, name in (
        ("relative_error_slow", "#0072B2", "o", "From numerical fast time"),
        ("relative_error_interval", "#D55E00", "s", r"From $\eta=0.82$"),
    ):
        means, maxima, positions = [], [], []
        for position, eta in enumerate(ascending):
            values = np.array([r[metric] for r in rows if r["eta"] == eta])
            values = values[np.isfinite(values)]
            if values.size:
                positions.append(position)
                means.append(100 * values.mean())
                maxima.append(100 * values.max())
        axes[1].plot(positions, means, marker=marker, color=color, linestyle="-", label=name + " (mean)")
        axes[1].plot(positions, maxima, marker=marker, color=color, linestyle="--", label=name + " (max)")
    axes[1].set(xlabel=r"Target polar angle $\eta$", ylabel="Relative error (%)", xlim=(-0.15, 4.15))
    axes[1].set_xticks(np.arange(len(ascending)), [f"{eta:.2f}" for eta in ascending])
    axes[1].legend(loc="upper left", fontsize=10)
    for ax, label in zip(axes, ("(a)", "(b)")):
        format_axes(ax)
        ax.text(0.5, -0.25, label, transform=ax.transAxes, ha="center", va="top", fontsize=15)
    save_plot(fig, "fig_targets_and_errors")


def save_plot(fig, stem):
    for extension in ("pdf", "png"):
        path = OUTPUT / "figures" / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}", flush=True)
    plt.close(fig)


def self_check():
    coefficient, phi0 = 0.01, 0.85
    ts = np.linspace(0, 150, 15001)
    phi = np.arctan(np.tan(phi0) * np.exp(-coefficient * ts))
    for eta in ETAS:
        measured = interpolated_hitting_time(ts, phi, float(eta))
        predicted = float((logtan(phi0) - logtan(eta)) / coefficient)
        np.testing.assert_allclose(measured, predicted, rtol=1e-7, atol=1e-7)
    assert np.isnan(interpolated_hitting_time(ts[:10], phi[:10], 0.30))
    assert first_threshold_index(np.array([2.0, 1.0]), 0.5) is None
    reference = interpolated_hitting_time(ts, phi, 0.82)
    for eta in ETAS[ETAS < 0.82]:
        interval = interpolated_hitting_time(ts, phi, float(eta)) - reference
        np.testing.assert_allclose(coefficient * interval, logtan(0.82) - logtan(eta), atol=1e-8)
    print("PASS synthetic logarithmic hitting law, interval law, and unreached-threshold checks.")



def validate():
    with (OUTPUT / "data/hitting_results.csv").open(newline="") as stream:
        rows = [{k: (v == "True" if v in {"True", "False"} else float(v))
                 for k, v in r.items()} for r in csv.DictReader(stream)]
    assert len(rows) == 125
    assert len({(r["K"], r["sigma_omega"], r["eta"]) for r in rows}) == 125
    _, ref = reference_data()
    for index, (k, sigma) in enumerate(PAIRS):
        group = [r for r in rows if r["K"] == k and r["sigma_omega"] == sigma]
        assert sorted(r["eta"] for r in group) == sorted(ETAS)
        assert len({r["t_f_num"] for r in group}) == 1
        for r in group:
            assert r["threshold_reached"] and r["fast_threshold_reached"]
            predicted = r["t_f_num"] + r["log_factor"] / r["Lambda_K"]
            np.testing.assert_allclose(predicted, r["T_eta_pred"], rtol=1e-13)
            error = abs(r["T_eta_sim"]-predicted)/r["T_eta_sim"]
            np.testing.assert_allclose(error, r["relative_error_total"], rtol=1e-11, atol=1e-14)
        r = next(r for r in group if r["eta"] == .82)
        for name, key in (("T_eta_sim", "T_sim"), ("T_eta_pred", "T_pred"),
                          ("t_f_num", "tf_num"), ("phi_tf", "phi_tf"),
                          ("Lambda_K", "Lambda_K"), ("relative_error_total", "relative_error")):
            np.testing.assert_allclose(r[name], np.asarray(ref[key]).ravel()[index],
                                       rtol=1e-9, atol=REFERENCE_TOLERANCES[key])
    with np.load(OUTPUT / "data/results.npz", allow_pickle=False) as z:
        for key in rows[0]:
            np.testing.assert_allclose(z[key], [r[key] for r in rows],
                                       rtol=0, atol=0, equal_nan=True)
    with (OUTPUT / "data/eta_summary.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    for actual, expected in zip(saved, summary(rows)):
        for key, value in expected.items():
            np.testing.assert_allclose(float(actual[key]), value, rtol=1e-13, equal_nan=True)
    print("PASS 125 target records, derived predictions/errors, NPZ/CSV consistency, "
          "and 25 manuscript Figure 4 comparisons.")
    return rows


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Integrate all cases; resume local _work caches.")
    parser.add_argument("--self-check", action="store_true", help="Check the synthetic hitting-time law without simulation.")
    parser.add_argument("--output-dir", type=Path, help="Separate experiment output directory.")
    args = parser.parse_args()
    if args.output_dir:
        OUTPUT = args.output_dir.resolve()
    if args.self_check:
        self_check()
        return
    if not args.recompute:
        validate()
        return
    for sub in ("data", "_work/cache", "figures"):
        (OUTPUT / sub).mkdir(parents=True, exist_ok=True)
    reference_path, reference = reference_data()
    all_rows, checks, cases = [], [], []
    start = time.monotonic()
    for index, (k, sigma) in enumerate(PAIRS):
        print(f"START {index+1}/25 K={k:g} sigma={sigma:.2f}", flush=True)
        data, cached = load_or_simulate(index)
        comparison = regression(data, reference)
        if not comparison["passed"]:
            raise RuntimeError(f"Publication regression failed for case {index}.")
        checks.append(comparison)
        all_rows.extend(target_records(data))
        cases.append(data)
        print(f"DONE cached={cached}", flush=True)
    refinement_rows = []
    for index in (4, 20):
        refined, _ = load_or_simulate(index, refined=True)
        coarse_rows = [r for r in all_rows if r["case_index"] == index]
        for coarse, fine in zip(coarse_rows, target_records(refined)):
            refinement_rows.append({
                "K": coarse["K"], "sigma_omega": coarse["sigma_omega"], "eta": coarse["eta"],
                "t_f_num_fixed": coarse["t_f_num"],
                "T_sim_coarse": coarse["T_eta_sim"], "T_sim_refined": fine["T_eta_sim"],
                "absolute_difference": abs(coarse["T_eta_sim"]-fine["T_eta_sim"]),
                "relative_difference_slow": abs(coarse["T_eta_sim"]-fine["T_eta_sim"])/coarse["slow_time_sim"],
                "prediction_absolute_difference": abs(coarse["T_eta_pred"]-fine["T_eta_pred"]),
            })
    write_csv(OUTPUT / "data/interpolation_refinement.csv", refinement_rows)
    write_csv(OUTPUT / "data/hitting_results.csv", all_rows)
    write_csv(OUTPUT / "data/eta_summary.csv", summary(all_rows))
    regression_report = {"passed": all(c["passed"] for c in checks),
                         "all_fields_exact_equal": all(c["all_fields_exact_equal"] for c in checks),
                         "cases": checks, "reference_path": str(reference_path.relative_to(ROOT)),
                         "reference_sha256": digest(reference_path)}
    save_metadata(OUTPUT / "data/eta082_regression.json", regression_report)
    payload = {k: np.asarray([r[k] for r in all_rows]) for k in all_rows[0]}
    payload.update({"eta_values": ETAS, "K_values": K_VALUES, "sigma_values": SIGMA_VALUES,
                    "ts": cases[0]["ts"], "phi_bar": np.stack([c["phi_bar"] for c in cases]),
                    "E_lock": np.stack([c["E_lock"] for c in cases])})
    np.savez_compressed(OUTPUT / "data/results.npz", **payload)
    metadata = {**CONFIG, "python": sys.version, "jax": jax.__version__, "diffrax": diffrax.__version__,
                "numpy": np.__version__, "jax_x64": bool(jax.config.jax_enable_x64),
                "tf_definition": "First original saved time with E_lock <= 5*rho^2; no argmin fallback.",
                "hitting_definition": "First linearly interpolated crossing after the fixed numerical fast time.",
                "error_definitions": ERROR_DEFINITIONS,
                "refinement_definition": "Half saved-time spacing at unchanged solver tolerances and fixed coarse fast times.",
                "eta082_regression": regression_report,
                "all_targets_reached": all(r["threshold_reached"] for r in all_rows),
                "case_count": len(cases), "target_record_count": len(all_rows),
                "sphere_norm_error_max": max(float(c["sphere_norm_error_max"]) for c in cases),
                "refinement_max_relative_difference_slow": max(r["relative_difference_slow"] for r in refinement_rows),
                "wall_seconds_this_invocation": time.monotonic()-start}
    save_metadata(OUTPUT / "data/metadata.json", metadata)
    validate()


if __name__ == "__main__":
    main()

