"""
Common utilities for numerical experiments in the common-axis sphere
Kuramoto/Lohe model.

The implementation uses JAX + Diffrax for time integration and NumPy/SciPy
for small nonlinear algebra tasks such as computing the classical locked
profile.

Shape convention:
    x0.shape == (N, 3)
    xs.shape == (T, N, 3)
    theta.shape == (T, N)
    phi.shape == (T, N)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import diffrax
except Exception as exc:  # pragma: no cover
    jax = None
    jnp = None
    diffrax = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


@dataclass
class SimulationResult:
    ts: np.ndarray
    xs: np.ndarray
    stats: Dict[str, float]


def ensure_jax_diffrax() -> None:
    """Raise a clear error if JAX or Diffrax is unavailable."""
    if _JAX_IMPORT_ERROR is not None:
        raise ImportError(
            "JAX and Diffrax are required for these experiments. "
            "Install them before running the figure scripts."
        ) from _JAX_IMPORT_ERROR


def make_figure_dir(path: str | Path = "figures") -> Path:
    """Create and return the directory used for figures."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_to_jsonable(value: Any) -> Any:
    """Convert common NumPy/Python objects to JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _metadata_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_jsonable(v) for v in value]
    return value


def save_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """
    Save experiment metadata as pretty-printed JSON.

    NumPy arrays and scalar types are converted to plain Python objects so
    that the file can be read without the numerical stack.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _metadata_to_jsonable(metadata)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_metadata(path: str | Path) -> Dict[str, Any]:
    """Load experiment metadata saved by save_metadata."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parameter_summary(
    *,
    N: int | None = None,
    K: float | None = None,
    omega: np.ndarray | None = None,
    omega_bar: float | None = None,
    sigma_omega: float | None = None,
    rho: float | None = None,
    theta0: float | None = None,
    phi0: float | None = None,
    perturb_scale: float | None = None,
    solver: str = "Diffrax Dopri5",
    rtol: float | None = None,
    atol: float | None = None,
    dt0: float | None = None,
    max_steps: int | None = None,
    t0: float | None = None,
    t1: float | None = None,
    num_save: int | None = None,
    locked_profile_residual: float | None = None,
    sphere_norm_error: float | None = None,
    t_f_num: float | None = None,
    C_tol: float | None = None,
    Lambda_K: float | None = None,
    figure_filename: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a standard metadata dictionary for the numerical experiments.

    If omega is supplied, missing frequency statistics are computed from it.
    Values left as None are omitted from the returned dictionary.
    """
    metadata: Dict[str, Any] = {}

    if omega is not None:
        omega_np = np.asarray(omega, dtype=float)
        if N is None:
            N = int(len(omega_np))
        if omega_bar is None:
            omega_bar = float(np.mean(omega_np))
        delta = omega_np - float(omega_bar)
        if sigma_omega is None:
            sigma_omega = float(np.sqrt(np.mean(delta**2)))
        if rho is None and K is not None:
            rho = float(np.max(np.abs(delta)) / float(K))

    entries = {
        "N": N,
        "K": K,
        "omega_bar": omega_bar,
        "sigma_omega": sigma_omega,
        "rho": rho,
        "theta0": theta0,
        "phi0": phi0,
        "perturb_scale": perturb_scale,
        "solver": solver,
        "rtol": rtol,
        "atol": atol,
        "dt0": dt0,
        "max_steps": max_steps,
        "t0": t0,
        "t1": t1,
        "num_save": num_save,
        "locked_profile_residual": locked_profile_residual,
        "sphere_norm_error": sphere_norm_error,
        "t_f_num": t_f_num,
        "C_tol": C_tol,
        "Lambda_K": Lambda_K,
        "figure_filename": figure_filename,
    }
    metadata.update({key: value for key, value in entries.items() if value is not None})

    if sigma_omega is not None and K is not None:
        metadata["sigma_omega_squared_over_K"] = float(sigma_omega) ** 2 / float(K)

    if extra:
        metadata.update(dict(extra))

    return _metadata_to_jsonable(metadata)


def deterministic_frequencies(
    N: int,
    omega_bar: float = 0.5,
    sigma_omega: float = 0.2,
) -> np.ndarray:
    """
    Deterministic mean-zero, variance-one sampling of frequency deviations.

    The returned frequencies satisfy approximately
        mean(omega) = omega_bar,
        mean((omega - omega_bar)**2) = sigma_omega**2.
    """
    xi = np.linspace(-1.0, 1.0, N)
    xi = xi - np.mean(xi)
    xi = xi / np.sqrt(np.mean(xi**2))
    return omega_bar + sigma_omega * xi


def gaussian_frequencies(
    N: int,
    omega_bar: float = 0.5,
    sigma_omega: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Random Gaussian frequency sampling with exact zero mean and unit variance."""
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=N)
    xi = xi - np.mean(xi)
    xi = xi / np.sqrt(np.mean(xi**2))
    return omega_bar + sigma_omega * xi


def frequency_statistics(omega: np.ndarray, K: float) -> Dict[str, float]:
    """Return omega_bar, sigma_omega, max frequency spread, and rho."""
    omega = np.asarray(omega, dtype=float)
    omega_bar = float(np.mean(omega))
    delta = omega - omega_bar
    sigma_omega = float(np.sqrt(np.mean(delta**2)))
    max_delta = float(np.max(np.abs(delta)))
    rho = max_delta / float(K)
    return {
        "omega_bar": omega_bar,
        "sigma_omega": sigma_omega,
        "max_delta": max_delta,
        "rho": rho,
    }


def locked_profile_residual(theta: np.ndarray, delta: np.ndarray, K: float) -> np.ndarray:
    """
    Residual of the classical Kuramoto locked-profile equation.

    Equation:
        0 = delta_i + K/N sum_k sin(theta_k - theta_i).
    """
    theta = np.asarray(theta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    N = len(theta)
    gaps = theta[None, :] - theta[:, None]
    return delta + (K / N) * np.sum(np.sin(gaps), axis=1)


def solve_locked_profile(
    omega: np.ndarray,
    K: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Compute the mean-zero classical Kuramoto locked profile.

    The unknown is solved on the mean-zero subspace by eliminating the last
    component.  The initial guess is the strong-coupling approximation
    theta_i = delta_i / K.

    Returns:
        theta: mean-zero locked profile
        residual_norm: max absolute residual of the full equation
    """
    omega = np.asarray(omega, dtype=float)
    N = len(omega)
    omega_bar = np.mean(omega)
    delta = omega - omega_bar

    # Initial guess on the mean-zero subspace.
    theta0 = delta / K
    theta0 = theta0 - np.mean(theta0)
    z0 = theta0[:-1].copy()

    def unpack(z):
        theta = np.empty(N, dtype=float)
        theta[:-1] = z
        theta[-1] = -np.sum(z)
        return theta

    def f_reduced(z):
        theta = unpack(z)
        return locked_profile_residual(theta, delta, K)[:-1]

    try:
        from scipy.optimize import root

        sol = root(f_reduced, z0, method="hybr", options={"maxfev": max_iter * (N + 1)})
        if sol.success:
            theta = unpack(sol.x)
        else:
            theta = theta0.copy()
            # fallback Newton-like fixed-point correction
            for _ in range(max_iter):
                r = locked_profile_residual(theta, delta, K)
                theta = theta + r / K
                theta = theta - np.mean(theta)
                if np.max(np.abs(r)) < tol:
                    break
    except Exception:
        theta = theta0.copy()
        for _ in range(max_iter):
            r = locked_profile_residual(theta, delta, K)
            theta = theta + r / K
            theta = theta - np.mean(theta)
            if np.max(np.abs(r)) < tol:
                break

    theta = theta - np.mean(theta)
    res = locked_profile_residual(theta, delta, K)
    return theta, float(np.max(np.abs(res)))


def compute_lambda_K(vartheta: np.ndarray, K: float) -> float:
    """Compute Lambda_K from the locked profile."""
    vartheta = np.asarray(vartheta, dtype=float)
    gaps = vartheta[:, None] - vartheta[None, :]
    return float(K * np.mean(1.0 - np.cos(gaps)))


def angles_to_sphere(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to points on S^2."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def sphere_to_angles(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an array of S^2 states to unwrapped theta and polar phi.

    Args:
        xs: array with shape (T, N, 3)

    Returns:
        theta: unwrapped azimuthal angle with shape (T, N)
        phi: polar angle with shape (T, N)
    """
    xs = np.asarray(xs)
    validate_sphere_trajectory_shape(xs, dim=3)
    theta_raw = np.arctan2(xs[..., 1], xs[..., 0])
    theta = np.unwrap(theta_raw, axis=0)
    z = np.clip(xs[..., 2], -1.0, 1.0)
    phi = np.arccos(z)
    return theta, phi


def mean_center(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Subtract the mean along the given axis."""
    return arr - np.mean(arr, axis=axis, keepdims=True)


def validate_sphere_trajectory_shape(xs: np.ndarray, dim: int = 3) -> Tuple[int, int, int]:
    """
    Validate the standard trajectory shape (T, N, dim).

    Returns:
        (T, N, dim) as integers.
    """
    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"Expected xs with shape (T, N, {dim}); got ndim={xs.ndim}.")
    if xs.shape[-1] != dim:
        raise ValueError(f"Expected last axis size {dim}; got shape={xs.shape}.")
    return int(xs.shape[0]), int(xs.shape[1]), int(xs.shape[2])

def sphere_rhs(t, y, args):
    """
    JAX RHS for the 3D axis-aligned common-axis sphere Kuramoto model.

    y is flattened with shape (N * 3,).
    args = (omega, K, N)
    """
    omega, K, N = args
    x = y.reshape((N, 3))
    mean_x = jnp.mean(x, axis=0)
    dot_mean = jnp.sum(x * mean_x[None, :], axis=1, keepdims=True)
    coupling = K * (mean_x[None, :] - dot_mean * x)

    # e3 cross x = (-x_2, x_1, 0)
    e3_cross_x = jnp.stack([-x[:, 1], x[:, 0], jnp.zeros((N,))], axis=1)
    natural = omega[:, None] * e3_cross_x
    return (natural + coupling).reshape((-1,))


def integrate_sphere_model(
    omega: np.ndarray,
    K: float,
    x0: np.ndarray,
    t0: float,
    t1: float,
    num_save: int,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    dt0: float | None = None,
    max_steps: int = 200000,
) -> SimulationResult:
    """
    Integrate the 3D sphere model using Diffrax Dopri5.

    Returns:
        SimulationResult with saved times, states, and diagnostic stats.
    """
    ensure_jax_diffrax()

    omega_np = np.asarray(omega, dtype=float)
    x0_np = np.asarray(x0, dtype=float)
    N = x0_np.shape[0]

    if dt0 is None:
        dt0 = min(1e-3, max((t1 - t0) / 1000.0, 1e-5))

    omega_j = jnp.asarray(omega_np)
    y0 = jnp.asarray(x0_np.reshape((-1,)))

    term = diffrax.ODETerm(sphere_rhs)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num_save))
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(omega_j, float(K), int(N)),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    ts = np.asarray(sol.ts)
    xs = np.asarray(sol.ys).reshape((num_save, N, 3))
    norm_error = float(np.max(np.abs(np.linalg.norm(xs, axis=-1) - 1.0)))

    return SimulationResult(ts=ts, xs=xs, stats={"sphere_norm_error": norm_error})


def locking_diagnostics(
    ts: np.ndarray,
    xs: np.ndarray,
    omega_bar: float,
    vartheta: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute theta, phi, mean quantities, and locking errors.

    Shape convention:
        ts: shape (T,)
        xs: shape (T, N, 3)
        vartheta: shape (N,)

    Returns a dictionary containing:
        theta, phi, y, y_bar, a, phi_bar, E_theta, E_phi, E_lock.
    """
    T, N, _ = validate_sphere_trajectory_shape(xs, dim=3)
    ts = np.asarray(ts, dtype=float)
    if ts.shape != (T,):
        raise ValueError(f"Expected ts with shape ({T},); got {ts.shape}.")
    vartheta = np.asarray(vartheta, dtype=float)
    if vartheta.shape != (N,):
        raise ValueError(f"Expected vartheta with shape ({N},); got {vartheta.shape}.")

    theta, phi = sphere_to_angles(xs)
    y = theta - omega_bar * ts[:, None]
    y_bar = np.mean(y, axis=1)
    a = y - y_bar[:, None]
    phi_bar = np.mean(phi, axis=1)

    E_theta = np.max(np.abs(a - vartheta[None, :]), axis=1)
    E_phi = np.max(np.abs(phi - phi_bar[:, None]), axis=1)
    E_lock = E_theta + E_phi

    return {
        "theta": theta,
        "phi": phi,
        "y": y,
        "y_bar": y_bar,
        "a": a,
        "phi_bar": phi_bar,
        "E_theta": E_theta,
        "E_phi": E_phi,
        "E_lock": E_lock,
    }


def first_threshold_time(ts: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Return first time at which values <= threshold; return NaN if not reached."""
    idx = np.where(values <= threshold)[0]
    if len(idx) == 0:
        return float("nan")
    return float(ts[idx[0]])


def first_threshold_index(values: np.ndarray, threshold: float) -> int | None:
    """Return first index at which values <= threshold; return None if not reached."""
    idx = np.where(np.asarray(values) <= threshold)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def interpolated_hitting_time(
    ts: np.ndarray,
    values: np.ndarray,
    threshold: float,
    start_time: float = 0.0,
) -> float:
    """
    Return the first linearly interpolated time when values reach threshold.

    This is intended for decreasing hitting-time diagnostics such as
    phi_bar(t) reaching eta after the fast layer.  The crossing condition is
    values <= threshold.  If the threshold is not reached after start_time,
    NaN is returned.
    """
    ts = np.asarray(ts, dtype=float)
    values = np.asarray(values, dtype=float)
    if ts.ndim != 1 or values.ndim != 1:
        raise ValueError("ts and values must be one-dimensional arrays.")
    if ts.shape != values.shape:
        raise ValueError(f"ts and values must have the same shape; got {ts.shape} and {values.shape}.")
    if len(ts) == 0:
        return float("nan")

    start_idx = int(np.searchsorted(ts, start_time, side="left"))
    if start_idx >= len(ts):
        return float("nan")
    if values[start_idx] <= threshold:
        return float(ts[start_idx])

    crossing = np.where(values[start_idx + 1 :] <= threshold)[0]
    if len(crossing) == 0:
        return float("nan")

    idx = start_idx + 1 + int(crossing[0])
    t0, t1 = float(ts[idx - 1]), float(ts[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    if v1 == v0:
        return t1

    alpha = (float(threshold) - v0) / (v1 - v0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return t0 + alpha * (t1 - t0)


def logtan(phi: np.ndarray) -> np.ndarray:
    """
    Compute log(tan(phi)) for polar angles in radians.

    Values outside the open interval (0, pi/2) naturally produce inf or NaN;
    fitting helpers filter non-finite values.
    """
    phi = np.asarray(phi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.tan(phi))


def fit_logtan_slope(
    ts: np.ndarray,
    phi_bar: np.ndarray,
    t_start: float,
    phi_min: float,
    phi_max: float,
    t_end: float | None = None,
    min_points: int = 3,
) -> Tuple[float, float, np.ndarray]:
    """
    Fit log(tan(phi_bar(t))) = slope * t + intercept on a filtered interval.

    The fit mask keeps times t >= t_start, optionally t <= t_end, and polar
    angles satisfying phi_min <= phi_bar <= phi_max.  Non-finite log-tangent
    values are excluded.  If too few points remain, (NaN, NaN, mask) is
    returned.
    """
    ts = np.asarray(ts, dtype=float)
    phi_bar = np.asarray(phi_bar, dtype=float)
    if ts.ndim != 1 or phi_bar.ndim != 1:
        raise ValueError("ts and phi_bar must be one-dimensional arrays.")
    if ts.shape != phi_bar.shape:
        raise ValueError(f"ts and phi_bar must have the same shape; got {ts.shape} and {phi_bar.shape}.")

    y = logtan(phi_bar)
    mask = (ts >= t_start) & (phi_bar >= phi_min) & (phi_bar <= phi_max) & np.isfinite(y)
    if t_end is not None:
        mask &= ts <= t_end

    if int(np.sum(mask)) < int(min_points):
        return float("nan"), float("nan"), mask

    slope, intercept = np.polyfit(ts[mask], y[mask], deg=1)
    return float(slope), float(intercept), mask


def make_theorem_regime_initial_condition(
    omega: np.ndarray,
    K: float,
    vartheta: np.ndarray,
    theta0: float = 0.3,
    phi0: float = 0.85,
    perturb_scale: float = 0.30,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Construct theorem-regime initial data:
        theta_i(0) = theta0 + vartheta_i + epsilon_i^theta,
        phi_i(0)   = phi0 + epsilon_i^phi,
    with deterministic O(rho) perturbations.
    """
    omega = np.asarray(omega, dtype=float)
    N = len(omega)
    stats = frequency_statistics(omega, K)
    rho = stats["rho"]

    grid = np.arange(N)
    epshat_theta = np.sin(2.0 * np.pi * grid / N)
    epshat_phi = np.cos(2.0 * np.pi * grid / N)
    epshat_theta = epshat_theta - np.mean(epshat_theta)
    epshat_phi = epshat_phi - np.mean(epshat_phi)
    epshat_norm = float(np.max(np.abs(epshat_theta)) + np.max(np.abs(epshat_phi)))
    if epshat_norm == 0.0:
        raise ValueError("Degenerate deterministic perturbation pattern.")
    epshat_theta = epshat_theta / epshat_norm
    epshat_phi = epshat_phi / epshat_norm

    eps_theta = perturb_scale * rho * epshat_theta
    eps_phi = perturb_scale * rho * epshat_phi

    theta_init = theta0 + np.asarray(vartheta) + eps_theta
    phi_init = phi0 + eps_phi
    phi_init = np.clip(phi_init, 1e-3, np.pi / 2 - 1e-3)
    x0 = angles_to_sphere(theta_init, phi_init)

    meta = {
        "rho": rho,
        "theta0": theta0,
        "phi0": phi0,
        "perturb_scale": perturb_scale,
        "epshat_normalization": "max|epshat_theta| + max|epshat_phi| = 1",
        "max_epshat_theta": float(np.max(np.abs(epshat_theta))),
        "max_epshat_phi": float(np.max(np.abs(epshat_phi))),
        "max_epshat_sum": float(np.max(np.abs(epshat_theta)) + np.max(np.abs(epshat_phi))),
        "max_eps_theta": float(np.max(np.abs(eps_theta))),
        "max_eps_phi": float(np.max(np.abs(eps_phi))),
        "max_eps_sum": float(np.max(np.abs(eps_theta)) + np.max(np.abs(eps_phi))),
    }
    return x0, meta


def save_figure(fig, figure_dir: str | Path, stem: str, dpi: int = 300) -> Tuple[Path, Path]:
    """Save a Matplotlib figure as both PDF and PNG."""
    figure_dir = make_figure_dir(figure_dir)
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path
