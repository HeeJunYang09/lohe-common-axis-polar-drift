# Multi-Target Hitting-Time Validation

This experiment tests the logarithmic target-angle dependence of the
hitting-time law beyond the single target used in manuscript Figure 4.
The targets are `eta = 0.84, 0.82, 0.70, 0.50, 0.30` radians.
The value 0.82 is the manuscript regression point, 0.84 is closer to the
initial mean angle 0.85, and the lower values probe longer polar motion.
They are diagnostic targets, not an equally spaced sampling of angles.

## Numerical Protocol

| Setting | Value |
|---|---|
| Oscillators | N = 28 |
| Coupling K | 6, 7.5, 9, 10.5, 12 |
| Frequency standard deviation | 0.20, 0.24, 0.28, 0.32, 0.36 |
| Frequency rule | Deterministic, as in Figure 4; mean 0.5 |
| Initial base angles | theta0 = 0.3, phi0 = 0.85 radians |
| Initial perturbation coefficient | c_init = 0.30 |
| Fast threshold | E_lock <= 5 rho^2 |
| Integrator | Diffrax Dopri5, JAX x64 |
| Tolerances | rtol = 1e-9, atol = 1e-11 |

The first integration retains Figure 4's interval [0,300] with 2200 saved
times. A continuation from the unmodified final state extends to t=450
with 1101 saved times, including the shared endpoint. No state
renormalization is applied at the join.

For each of the 25 trajectories, t_f is the first original saved time
satisfying the fast threshold. The same t_f and interpolated phi_bar(t_f)
are used for all five targets. Missing locking is an error, not replaced by
a minimum-residual time. The first crossing of phi_bar <= eta after t_f
is evaluated by linear interpolation.

The prediction is

```text
T_pred = t_f + log(tan(phi_bar(t_f)) / tan(eta)) / Lambda_K.
```

Lambda_K comes from the computed locked profile, not a fit to hitting times.

## Error Definitions and Results

The files distinguish three errors, whose denominators must not be confused.

- `relative_error_total = abs(T_sim - T_pred) / T_sim`, as in Figure 4.
- `relative_error_slow = abs(T_sim - T_pred) / (T_pred - t_f)`.
- `relative_error_interval = abs(delta_T_sim - delta_T_pred) / delta_T_pred`,
  with travel times measured from eta=0.82 to each lower target.

All 125 target comparisons reach their targets by t=450.

| eta | Mean total-time error (%) | Maximum total-time error (%) |
|---|---|---|
| 0.84 | 1.0443 | 4.2966 |
| 0.82 | 0.3825 | 1.5614 |
| 0.70 | 0.1186 | 0.4403 |
| 0.50 | 0.0927 | 0.3126 |
| 0.30 | 0.0954 | 0.3055 |

The eta=0.82 results reproduce all 25 manuscript comparison records exactly
in the stored regression. Independent travel intervals from eta=0.82 remove
the fast-time and initial-angle anchors from the logarithmic comparison.

Smaller relative errors do not necessarily imply smaller absolute errors.
For K=6 and sigma_omega=0.36, T_sim increases from 0.9702 to 60.2415
between eta=0.84 and 0.30, while the absolute error increases from 0.0417
to 0.1840. The relative errors are not strictly monotone in eta.
These tests do not establish a uniform eta-to-zero bound or a pole
convergence rate. The five target observations along a trajectory are
correlated, not independent random trials.

## Files and Figures

- `data/hitting_results.csv` contains 125 individual target records.
- `data/eta_summary.csv` aggregates all three error definitions by target.
- `data/results.npz` contains the same records and saved mean-angle and
  locking-error traces. Shared time has shape (3300,), traces (25,3300).
- `data/metadata.json` records numerical settings and diagnostic definitions.
- `data/eta082_regression.json` records comparisons with Figure 4.
- `data/interpolation_refinement.csv` compares two representative cases
  after halving the saved-time spacing, with coarse t_f anchors held fixed.
  The largest relative change in slow duration is about 2.59e-6.
  This is an interpolation check, not a solver-tolerance refinement.
- `figures/fig_multi_target_hitting_time.pdf` compares scaled travel times
  with the logarithmic law and shows mean/maximum total-time errors.
- `figures/fig_logarithmic_law.pdf` also shows travel intervals from eta=0.82.
- `figures/fig_targets_and_errors.pdf` shows representative arrival times
  and the slow-time and interval error definitions.

Angle categories in the error panels are equally spaced for readability;
this is not a linear angle axis.

## Commands

From the repository root, validate and render the included results without
simulation.

```bash
conda run -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/run_multi_target.py
conda run -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/plot_multi_target.py
conda run -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/run_multi_target.py --self-check
```

To recompute all 25 trajectories and the two saved-grid refinements in a
separate directory, use

```bash
conda run --no-capture-output -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/run_multi_target.py --recompute --output-dir /tmp/lohe-multi-target-recompute
```

Omit `--output-dir` only to intentionally replace this experiment's
distributed data. Recalculation can resume compatible caches under
`_work/cache/`. Runtime depends on the JAX backend; the command above
performs new integrations rather than just regenerating figures.

