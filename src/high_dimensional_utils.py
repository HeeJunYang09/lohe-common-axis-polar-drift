"""Utilities for the five-dimensional common-axis block experiment."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import brentq

try:
    import jax.numpy as jnp
    import diffrax
except Exception as exc:  # pragma: no cover
    jnp = None
    diffrax = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


def ensure_jax_diffrax() -> None:
    """Raise a clear error if JAX or Diffrax is unavailable."""
    if _JAX_IMPORT_ERROR is not None:
        raise ImportError("JAX and Diffrax are required for the d=5 experiment.") from _JAX_IMPORT_ERROR


def make_rotation_generator(dtype=np.float64) -> np.ndarray:
    """Return J = [[0,-1],[1,0]] with shape (2,2)."""
    return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)


def _as_pair(values: Tuple[float, float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}.")
    return arr


def make_d5_frequency_matrices(omega_blocks: np.ndarray) -> np.ndarray:
    """Return block-diagonal d=5 frequency matrices from block frequencies."""
    omega_blocks = np.asarray(omega_blocks, dtype=float)
    if omega_blocks.ndim != 2 or omega_blocks.shape[1] != 2:
        raise ValueError(f"omega_blocks must have shape (N,2), got {omega_blocks.shape}.")
    N = omega_blocks.shape[0]
    J = make_rotation_generator(dtype=omega_blocks.dtype)
    mats = np.zeros((N, 5, 5), dtype=omega_blocks.dtype)
    mats[:, 0:2, 0:2] = omega_blocks[:, 0, None, None] * J[None, :, :]
    mats[:, 2:4, 2:4] = omega_blocks[:, 1, None, None] * J[None, :, :]
    return mats


def validate_common_axis_structure(
    omega_matrices: np.ndarray,
    axis: np.ndarray | None = None,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """Return maximum skew, axis, and pairwise commutator residuals."""
    omega_matrices = np.asarray(omega_matrices, dtype=float)
    if omega_matrices.ndim != 3 or omega_matrices.shape[1:] != (5, 5):
        raise ValueError(f"omega_matrices must have shape (N,5,5), got {omega_matrices.shape}.")
    if axis is None:
        axis = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (5,):
        raise ValueError(f"axis must have shape (5,), got {axis.shape}.")

    skew = float(np.max(np.linalg.norm(omega_matrices + np.swapaxes(omega_matrices, 1, 2), axis=(1, 2))))
    axis_res = float(np.max(np.linalg.norm(np.einsum("nij,j->ni", omega_matrices, axis), axis=1)))
    comm_res = 0.0
    for i in range(omega_matrices.shape[0]):
        for j in range(i + 1, omega_matrices.shape[0]):
            comm = omega_matrices[i] @ omega_matrices[j] - omega_matrices[j] @ omega_matrices[i]
            comm_res = max(comm_res, float(np.linalg.norm(comm, ord=2)))
    diagnostics = {
        "max_skew_residual": skew,
        "max_axis_residual": axis_res,
        "max_commutator_residual": comm_res,
    }
    bad = {key: value for key, value in diagnostics.items() if value > atol}
    if bad:
        raise ValueError(f"common-axis structure validation failed at atol={atol}: {bad}")
    return diagnostics


def standardized_sample(x: np.ndarray) -> np.ndarray:
    """Recenter to zero empirical mean and rescale to unit empirical variance."""
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    variance = float(np.mean(centered**2))
    if variance <= 0.0:
        raise ValueError("cannot standardize a zero-variance sample.")
    return centered / np.sqrt(variance)


def make_deterministic_two_block_samples(
    N: int,
    orthogonalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return deterministic standardized two-block samples with shape (N,2)."""
    if N < 4:
        raise ValueError("N must be at least 4 for deterministic two-block samples.")
    idx = np.arange(N, dtype=float)
    block1 = standardized_sample(np.linspace(-1.0, 1.0, N))
    block2 = standardized_sample(np.cos(2.0 * np.pi * (idx + 0.5) / float(N)))
    corr = float(np.mean(block1 * block2))
    did_orthogonalize = False
    if orthogonalize and abs(corr) > 0.1:
        block2 = block2 - np.mean(block2 * block1) * block1
        block2 = standardized_sample(block2)
        corr = float(np.mean(block1 * block2))
        did_orthogonalize = True
    xi = np.column_stack([block1, block2])
    diagnostics = {
        "mean_block1": float(np.mean(xi[:, 0])),
        "mean_block2": float(np.mean(xi[:, 1])),
        "variance_block1": float(np.mean(xi[:, 0] ** 2)),
        "variance_block2": float(np.mean(xi[:, 1] ** 2)),
        "correlation": corr,
        "orthogonalized": float(did_orthogonalize),
    }
    return xi, diagnostics


def make_gaussian_two_block_samples(
    N: int,
    seed: int,
    correlated: bool = False,
) -> np.ndarray:
    """Return standardized Gaussian samples with shape (N,2)."""
    rng = np.random.default_rng(int(seed))
    if correlated:
        raw = rng.normal(size=N)
        xi = np.column_stack([standardized_sample(raw), standardized_sample(raw)])
    else:
        xi = np.column_stack([standardized_sample(rng.normal(size=N)), standardized_sample(rng.normal(size=N))])
    return xi


def make_two_block_frequency_package(
    N: int,
    sigma_std_target: Tuple[float, float],
    bar_omega_target: Tuple[float, float],
    mode: str,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Create realized d=5 two-block frequencies and their empirical moments."""
    sigma = _as_pair(sigma_std_target, "sigma_std_target")
    means = _as_pair(bar_omega_target, "bar_omega_target")
    if np.any(sigma < 0.0):
        raise ValueError("sigma_std_target entries must be nonnegative.")
    modes = {
        "deterministic_independent",
        "deterministic_correlated",
        "gaussian_independent",
        "gaussian_correlated",
    }
    if mode not in modes:
        raise ValueError(f"unknown frequency mode {mode!r}.")
    orthogonalized = False
    if mode.startswith("deterministic"):
        if seed is not None:
            raise ValueError("deterministic frequency modes require seed is None.")
        if mode == "deterministic_independent":
            xi, diag = make_deterministic_two_block_samples(N)
            orthogonalized = bool(diag["orthogonalized"])
        else:
            base = standardized_sample(np.linspace(-1.0, 1.0, N))
            xi = np.column_stack([base, base])
    else:
        if seed is None:
            raise ValueError("Gaussian frequency modes require an integer seed.")
        xi = make_gaussian_two_block_samples(N, seed, correlated=(mode == "gaussian_correlated"))

    omega_blocks = means[None, :] + sigma[None, :] * xi
    bar_emp = np.mean(omega_blocks, axis=0)
    delta_blocks = omega_blocks - bar_emp[None, :]
    variance_emp = np.mean(delta_blocks**2, axis=0)
    sigma_emp = np.sqrt(variance_emp)
    omega_matrices = make_d5_frequency_matrices(omega_blocks)
    bar_matrix = make_d5_frequency_matrices(bar_emp[None, :])[0]
    delta_matrices = omega_matrices - bar_matrix[None, :, :]
    delta_trans = delta_matrices[:, 0:4, 0:4]
    Q_transverse = -np.mean(np.einsum("nij,njk->nik", delta_trans, delta_trans), axis=0)
    Q_expected = np.diag(np.repeat(variance_emp, 2))
    Q_block_error = float(np.linalg.norm(Q_transverse - Q_expected, ord=2))
    if Q_block_error > 1e-12:
        raise ValueError(f"Q block error {Q_block_error:.3e} exceeds 1e-12.")
    return {
        "xi": xi,
        "omega_blocks": omega_blocks,
        "delta_blocks": delta_blocks,
        "bar_omega_target": means,
        "bar_omega_empirical": bar_emp,
        "sigma_std_target": sigma,
        "sigma_std_empirical": sigma_emp,
        "variance_empirical": variance_emp,
        "omega_matrices": omega_matrices,
        "bar_omega_matrix": bar_matrix,
        "delta_matrices": delta_matrices,
        "Q_transverse": Q_transverse,
        "Q_block_error": Q_block_error,
        "sample_correlation": float(np.mean(xi[:, 0] * xi[:, 1])),
        "orthogonalized": bool(orthogonalized),
        "rho_numerator": float(np.max(np.abs(delta_blocks))),
    }


def make_common_transverse_direction(
    block_weights: Tuple[float, float],
    block_angles: Tuple[float, float],
) -> np.ndarray:
    """Return a unit common transverse direction in R^4."""
    weights = _as_pair(block_weights, "block_weights")
    angles = _as_pair(block_angles, "block_angles")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise ValueError("block_weights must be positive and sum to one.")
    q1 = np.array([np.cos(angles[0]), np.sin(angles[0]), 0.0, 0.0])
    q2 = np.array([0.0, 0.0, np.cos(angles[1]), np.sin(angles[1])])
    u_star = np.sqrt(weights[0]) * q1 + np.sqrt(weights[1]) * q2
    return u_star / np.linalg.norm(u_star)


def make_generic_d5_perturbations(N: int, u_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean-zero generic tangent and polar perturbations."""
    u_star = np.asarray(u_star, dtype=float)
    if u_star.shape != (4,):
        raise ValueError(f"u_star must have shape (4,), got {u_star.shape}.")
    idx = np.arange(N, dtype=float)
    raw = np.column_stack(
        [
            np.sin(2.0 * np.pi * idx / N),
            np.cos(4.0 * np.pi * idx / N),
            np.sin(6.0 * np.pi * idx / N),
            np.cos(8.0 * np.pi * idx / N),
        ]
    )
    projector = np.eye(4) - np.outer(u_star, u_star)
    h = raw @ projector.T
    h = h - np.mean(h, axis=0, keepdims=True)
    h = h @ projector.T
    h_max = float(np.max(np.linalg.norm(h, axis=1)))
    if h_max <= 1e-14:
        raise ValueError("generic transverse perturbation has negligible H_max.")
    zeta_u_pre = h / h_max
    zeta_phi = np.sin(2.0 * np.pi * (idx + 1.0 / 3.0) / N)
    zeta_phi = zeta_phi - np.mean(zeta_phi)
    joint_scale = float(np.max(np.linalg.norm(zeta_u_pre, axis=1)) + np.max(np.abs(zeta_phi)))
    if joint_scale <= 1e-14:
        raise ValueError("generic perturbation joint scale is negligible.")
    zeta_u = zeta_u_pre / joint_scale
    zeta_phi = zeta_phi / joint_scale
    return zeta_u, zeta_phi


def make_d5_initial_state(
    delta_matrices: np.ndarray,
    K: float,
    phi0: float,
    block_weights: Tuple[float, float],
    block_angles: Tuple[float, float],
    rho: float,
    c_init: float,
) -> Dict[str, np.ndarray]:
    """Construct theorem-regime initial data for the d=5 experiment."""
    delta_matrices = np.asarray(delta_matrices, dtype=float)
    if delta_matrices.ndim != 3 or delta_matrices.shape[1:] != (5, 5):
        raise ValueError(f"delta_matrices must have shape (N,5,5), got {delta_matrices.shape}.")
    N = delta_matrices.shape[0]
    u_star0 = make_common_transverse_direction(block_weights, block_angles)
    zeta_u, zeta_phi = make_generic_d5_perturbations(N, u_star0)
    delta4 = delta_matrices[:, 0:4, 0:4]
    correction = np.einsum("nij,j->ni", delta4, u_star0) / float(K)
    v = u_star0[None, :] + correction + float(c_init) * float(rho) * zeta_u
    v_norm = np.linalg.norm(v, axis=1)
    if np.any(v_norm <= 1e-14):
        raise ValueError("initial transverse vector normalization failed.")
    u0 = v / v_norm[:, None]
    phi_i = float(phi0) + float(c_init) * float(rho) * zeta_phi
    if not (0.70 < float(np.min(phi_i)) < float(np.max(phi_i)) < 1.00):
        raise ValueError(
            "initial latitude check failed: "
            f"K={K}, rho={rho}, min_phi={np.min(phi_i)}, max_phi={np.max(phi_i)}"
        )
    x0 = np.empty((N, 5), dtype=float)
    x0[:, 0:4] = np.sin(phi_i)[:, None] * u0
    x0[:, 4] = np.cos(phi_i)
    max_u_error = float(np.max(np.abs(np.linalg.norm(u0, axis=1) - 1.0)))
    max_x_error = float(np.max(np.abs(np.linalg.norm(x0, axis=1) - 1.0)))
    if max_u_error > 1e-13 or max_x_error > 1e-13 or np.min(x0[:, 4]) <= 0.0:
        raise ValueError(
            "initial norm or hemisphere assertion failed: "
            f"max_u_error={max_u_error:.3e}, max_x_error={max_x_error:.3e}, min_z={np.min(x0[:,4]):.3e}"
        )
    return {
        "x0": x0,
        "u0": u0,
        "phi0_i": phi_i,
        "u_star0": u_star0,
        "zeta_u": zeta_u,
        "zeta_phi": zeta_phi,
    }


def d5_rhs(t, x_flat, args):
    """JAX-compatible RHS for the full d=5 sphere Kuramoto model."""
    omega_matrices, K, N = args
    x = x_flat.reshape((N, 5))
    mean_x = jnp.mean(x, axis=0)
    natural = jnp.einsum("nij,nj->ni", omega_matrices, x)
    projection = jnp.sum(x * mean_x[None, :], axis=1, keepdims=True)
    coupling = K * (mean_x[None, :] - projection * x)
    return (natural + coupling).reshape((-1,))


def integrate_d5(
    x0: np.ndarray,
    omega_matrices: np.ndarray,
    K: float,
    ts: np.ndarray,
    *,
    rtol: float,
    atol: float,
    dt0: float,
    max_steps: int,
) -> Dict[str, Any]:
    """Integrate the d=5 full model on a supplied save grid."""
    ensure_jax_diffrax()
    x0 = np.asarray(x0, dtype=float)
    ts = np.asarray(ts, dtype=float)
    omega_matrices = np.asarray(omega_matrices, dtype=float)
    N = x0.shape[0]
    term = diffrax.ODETerm(d5_rhs)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.asarray(ts))
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=float(dt0),
        y0=jnp.asarray(x0.reshape(-1)),
        args=(jnp.asarray(omega_matrices), float(K), int(N)),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(max_steps),
    )
    xs = np.asarray(sol.ys).reshape((len(ts), N, 5))
    sphere_error = float(np.max(np.abs(np.linalg.norm(xs, axis=-1) - 1.0)))
    return {
        "ts": np.asarray(sol.ts),
        "x": xs,
        "sphere_error": sphere_error,
        "solver_stats": getattr(sol, "stats", {}),
    }


def cartesian_to_polar_d5(
    x: np.ndarray,
    eps_pole: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert d=5 Cartesian states to polar angle and transverse direction."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 5:
        raise ValueError(f"x last dimension must be 5, got {x.shape}.")
    z = np.clip(x[..., 4], -1.0, 1.0)
    phi = np.arccos(z)
    sin_phi = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    denom = np.maximum(sin_phi, float(eps_pole))
    u = x[..., 0:4] / denom[..., None]
    return phi, u, sin_phi


def rotate_transverse_frame(
    u: np.ndarray,
    ts: np.ndarray,
    bar_omega_empirical: np.ndarray,
) -> np.ndarray:
    """Apply block rotations R(-bar_omega_alpha t) to transverse vectors."""
    u = np.asarray(u, dtype=float)
    ts = np.asarray(ts, dtype=float)
    bar = _as_pair(bar_omega_empirical, "bar_omega_empirical")
    if u.ndim != 3 or u.shape[-1] != 4:
        raise ValueError(f"u must have shape (T,N,4), got {u.shape}.")
    out = np.empty_like(u)
    for block in range(2):
        angle = -bar[block] * ts
        c = np.cos(angle)
        s = np.sin(angle)
        a = u[:, :, 2 * block]
        b = u[:, :, 2 * block + 1]
        out[:, :, 2 * block] = c[:, None] * a - s[:, None] * b
        out[:, :, 2 * block + 1] = s[:, None] * a + c[:, None] * b
    return out


def common_transverse_direction(u_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized common transverse direction and mean-vector norm."""
    u_rot = np.asarray(u_rot, dtype=float)
    mean_u = np.mean(u_rot, axis=1)
    mean_norm = np.linalg.norm(mean_u, axis=1)
    u_star = np.full_like(mean_u, np.nan)
    mask = mean_norm > 1e-14
    u_star[mask] = mean_u[mask] / mean_norm[mask, None]
    return u_star, mean_norm


def block_weights(u_star: np.ndarray) -> np.ndarray:
    """Return two block weights from a common transverse direction."""
    u_star = np.asarray(u_star, dtype=float)
    p1 = np.sum(u_star[..., 0:2] ** 2, axis=-1)
    p2 = np.sum(u_star[..., 2:4] ** 2, axis=-1)
    return np.stack([p1, p2], axis=-1)


def first_order_ansatz_residuals(
    u_rot: np.ndarray,
    u_star: np.ndarray,
    delta_matrices: np.ndarray,
    K: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return tangent and normalized Euclidean ansatz residuals."""
    u_rot = np.asarray(u_rot, dtype=float)
    u_star = np.asarray(u_star, dtype=float)
    delta4 = np.asarray(delta_matrices, dtype=float)[:, 0:4, 0:4]
    T = u_rot.shape[0]
    R_ans = np.full(T, np.nan)
    R_unit = np.full(T, np.nan)
    for n in range(T):
        us = u_star[n]
        if not np.all(np.isfinite(us)):
            continue
        P = np.eye(4) - np.outer(us, us)
        xi_sim = u_rot[n] @ P.T
        xi_pred = np.einsum("nij,j->ni", delta4, us) / float(K)
        R_ans[n] = float(np.max(np.linalg.norm(xi_sim - xi_pred, axis=1)))
        pred = us[None, :] + xi_pred
        pred = pred / np.linalg.norm(pred, axis=1)[:, None]
        R_unit[n] = float(np.max(np.linalg.norm(u_rot[n] - pred, axis=1)))
    return R_ans, R_unit


def transverse_diameter(u_rot: np.ndarray) -> np.ndarray:
    """Return the maximum pairwise transverse diameter at each saved time."""
    u_rot = np.asarray(u_rot, dtype=float)
    gram = np.einsum("tnd,tmd->tnm", u_rot, u_rot)
    dist2 = np.maximum(0.0, 2.0 - 2.0 * gram)
    return np.sqrt(np.max(dist2, axis=(1, 2)))


def exact_mean_polar_drift_coefficient(
    phi: np.ndarray,
    u_rot: np.ndarray,
    K: float,
    phi_bar: np.ndarray,
    eps_pole: float = 1e-10,
) -> np.ndarray:
    """Return the exact algebraic polar drift coefficient."""
    phi = np.asarray(phi, dtype=float)
    u_rot = np.asarray(u_rot, dtype=float)
    phi_bar = np.asarray(phi_bar, dtype=float)
    N = phi.shape[1]
    gram = np.einsum("tnd,tmd->tnm", u_rot, u_rot)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    weighted = (1.0 - gram) * cos_phi[:, :, None] * sin_phi[:, None, :]
    numerator = float(K) * np.sum(weighted, axis=(1, 2)) / float(N * N)
    denominator = np.sin(phi_bar) * np.cos(phi_bar)
    out = np.full_like(phi_bar, np.nan, dtype=float)
    mask = np.abs(denominator) >= float(eps_pole)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def find_persistent_fast_time(
    ts: np.ndarray,
    R_ans: np.ndarray,
    E_phi: np.ndarray,
    rho: float,
    K: float,
    C_tol: float,
    persistence_Kt: float,
    search_stop_index: int | None = None,
) -> Tuple[float, int, bool]:
    """Find the first persistent time satisfying the fast-layer threshold."""
    ts = np.asarray(ts, dtype=float)
    R_ans = np.asarray(R_ans, dtype=float)
    E_phi = np.asarray(E_phi, dtype=float)
    stop = len(ts) - 1 if search_stop_index is None else int(search_stop_index)
    stop = max(0, min(stop, len(ts) - 1))
    threshold = float(C_tol) * float(rho) ** 2
    for j in range(stop + 1):
        end_time = ts[j] + float(persistence_Kt) / float(K)
        k = int(np.searchsorted(ts, end_time, side="left"))
        if k > stop:
            break
        if np.all(R_ans[j : k + 1] <= threshold) and np.all(E_phi[j : k + 1] <= threshold):
            return float(ts[j]), int(j), True
    return float("nan"), -1, False


def reduced_block_weights(
    ts: np.ndarray,
    t_f: float,
    p_f: np.ndarray,
    variance_empirical: np.ndarray,
    K: float,
) -> np.ndarray:
    """Return the explicit reduced block weights."""
    ts = np.asarray(ts, dtype=float)
    p_f = _as_pair(p_f, "p_f")
    variance = _as_pair(variance_empirical, "variance_empirical")
    out = np.full((len(ts), 2), np.nan)
    mask = ts >= float(t_f)
    tau_t = (ts[mask] - float(t_f)) / float(K)
    logw = np.log(p_f[None, :]) - 2.0 * tau_t[:, None] * variance[None, :]
    logw = logw - np.max(logw, axis=1, keepdims=True)
    weights = np.exp(logw)
    out[mask] = weights / np.sum(weights, axis=1, keepdims=True)
    return out


def reduced_mean_polar_angle(
    ts: np.ndarray,
    t_f: float,
    phi_f: float,
    p_f: np.ndarray,
    variance_empirical: np.ndarray,
    K: float,
) -> np.ndarray:
    """Return the explicit reduced mean polar angle."""
    p_red = reduced_block_weights(ts, t_f, p_f, variance_empirical, K)
    variance = _as_pair(variance_empirical, "variance_empirical")
    ts = np.asarray(ts, dtype=float)
    out = np.full(len(ts), np.nan)
    mask = ts >= float(t_f)
    tau_t = (ts[mask] - float(t_f)) / float(K)
    S = np.sum(p_f[None, :] * np.exp(-2.0 * tau_t[:, None] * variance[None, :]), axis=1)
    out[mask] = np.arctan(np.tan(float(phi_f)) * np.sqrt(S))
    return out


def reduced_hitting_time_d5(
    t_f: float,
    phi_f: float,
    eta: float,
    p_f: np.ndarray,
    variance_empirical: np.ndarray,
    K: float,
) -> float:
    """Solve the d=5 reduced hitting-time equation by monotone bracketing."""
    p_f = _as_pair(p_f, "p_f")
    variance = _as_pair(variance_empirical, "variance_empirical")
    if not (0.0 < eta <= phi_f < 0.5 * np.pi):
        raise ValueError(f"require 0 < eta <= phi_f < pi/2, got eta={eta}, phi_f={phi_f}.")
    if np.isclose(eta, phi_f, rtol=0.0, atol=1e-14):
        return float(t_f)
    if not np.isclose(np.sum(p_f), 1.0, atol=1e-12) or np.any(p_f < -1e-12):
        raise ValueError("p_f must be nonnegative and sum to one.")
    if K <= 0.0 or np.any(variance < -1e-14):
        raise ValueError("K must be positive and variances nonnegative.")
    target = (np.tan(float(eta)) / np.tan(float(phi_f))) ** 2

    def F(tau: float) -> float:
        return float(np.sum(p_f * np.exp(-2.0 * variance * tau)) - target)

    limit = float(np.sum(p_f[variance <= 1e-15]) - target)
    if limit >= 0.0:
        raise ValueError("target is not reached at finite slow time.")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if F(hi) <= 0.0:
            tau = brentq(F, lo, hi)
            return float(t_f + float(K) * tau)
        hi *= 2.0
    raise RuntimeError(f"failed to bracket d=5 hitting time: F({hi})={F(hi)}.")


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Return masked relative L2 error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != y_true.shape:
        mask = np.broadcast_to(mask, y_true.shape)
    mask = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(mask)) == 0:
        return float("nan")
    num = float(np.sum((y_true[mask] - y_pred[mask]) ** 2))
    den = float(np.sum(y_true[mask] ** 2))
    return float(np.sqrt(num / max(den, 1e-28)))


def fit_log_weight_ratio_slope(
    ts: np.ndarray,
    p_sim: np.ndarray,
    t_f: float,
    K: float,
    variance_empirical: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Fit the simulated block-weight log-ratio against slow time."""
    ts = np.asarray(ts, dtype=float)
    p_sim = np.asarray(p_sim, dtype=float)
    variance = _as_pair(variance_empirical, "variance_empirical")
    valid = np.asarray(mask, dtype=bool) & np.all(p_sim > 0.0, axis=1)
    valid &= np.isfinite(ts) & np.all(np.isfinite(p_sim), axis=1)
    count = int(np.sum(valid))
    slope_pred = float(2.0 * (variance[1] - variance[0]))
    if count < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "slope_pred": slope_pred,
            "relative_error": float("nan"),
            "point_count": count,
        }
    tau = (ts[valid] - float(t_f)) / float(K)
    y_raw = np.log(p_sim[valid, 0] / p_sim[valid, 1])
    y = y_raw - y_raw[0]
    slope, intercept = np.polyfit(tau, y, deg=1)
    yhat = slope * tau + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    rel = abs(float(slope) - slope_pred) / max(abs(slope_pred), 1e-14)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "slope_pred": slope_pred,
        "relative_error": float(rel),
        "point_count": count,
    }
