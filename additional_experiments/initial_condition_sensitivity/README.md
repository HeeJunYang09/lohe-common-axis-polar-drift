# Sensitivity to Initial Conditions

This experiment changes initialization while holding the manuscript
parameter grids, frequency samples, solvers, tolerances, and diagnostic
windows fixed. It tests whether a wider family of initial states exhibits
fast relaxation followed by the predicted slow motion.

## Initial-Condition Groups

- A uses the original manuscript initialization.
- B removes the locked-profile correction in three dimensions and the
  first-order transverse correction before normalization in five dimensions.
  The original small perturbations are retained.
- C removes these corrections and uses random perturbations of amplitude
  a = 0.05, 0.15, 0.30, independent of rho.

In three dimensions, group C uses

```text
theta_i(0) = 0.3 + a * zeta_theta_i
phi_i(0)   = 0.85 + a * zeta_phi_i.
```

Two independent uniform[-1,1] vectors are drawn, centered, and separately
normalized to maximum absolute value one. The draws are independent of
the deterministic frequencies.

In five dimensions,

```text
u_i(0)   = normalize(u_star(0) + a * h_i)
phi_i(0) = 0.85 + a * zeta_phi_i.
```

The h_i are perpendicular to u_star(0), have maximum norm one, and occur
in opposite pairs with randomized particle assignments. Tangent directions
are obtained from projected Gaussian draws, with uniform radii in [0.1,1]
before maximum normalization. Opposite pairs preserve the mean transverse
direction after normalization. The polar perturbations use the same
centering and normalization as in three dimensions.

For each dimension and amplitude, seeds 0--4 specify five realizations.
Each realization is reused across the parameter grid. These are paired
diagnostics, not independent estimates of basin probabilities.
All initial polar angles lie in [0.55,1.15], inside the same open hemisphere.
The five-dimensional common direction has equal initial block weights.

## Fixed Settings and Diagnostics

| Setting | Three dimensions | Five dimensions |
|---|---|---|
| Manuscript comparison | Figure 3(b) | Figure 6 |
| N | 32 | 64 |
| K | 8, 10, 12 | 8, 10, 12, 16 |
| Frequency spread | sigma_omega = 0.16, 0.20, 0.24 | (sigma1,sigma2) = (0.10,0.30) |
| Mean frequency | 0.5 | (0.5,-0.25) |
| Saved times | 1800 on [0,220] | Original 3000-point grid up to 1+18K |
| Slow diagnostic | log tan(phi_bar) slope | log(p1/p2) slope in tau=(t-t_f)/K |
| Polar comparison window | 0.22 <= phi_bar <= 0.78 | 0.35 <= phi_bar <= 0.80 |

All integrations use the publication JAX x64 / Diffrax solver settings,
rtol=1e-9 and atol=1e-11. The same root helpers are reused.

In three dimensions, t_f is the first saved time with E_lock <= 5 rho^2.
In five dimensions, both R_ans and E_phi must remain <= 5 rho^2 throughout
the saved persistence window of length 2/K. Threshold maintenance over the
rest of the saved trajectory is checked separately. Zero t_f means the
initial saved state already meets the criterion; it does not mean that
no fast transient exists.

The measured 3D drift rate is minus the physical-time logtan slope.
Its prediction Lambda_K is computed from the locked profile. The 5D
block-selection slope is predicted to be 2(sigma2^2-sigma1^2)=0.16.
Reduced polar predictions are anchored to the observed post-fast state.
The coefficients are not adjusted to fit the trajectories.

## Results

All 153 three-dimensional and 68 five-dimensional cases attain and
subsequently maintain their thresholds at saved times. The table reports
maximum absolute relative rate errors as percentages.

| Group | 3D Kt_f range | 3D max error (%) | 5D Kt_f range | 5D max error (%) |
|---|---|---|---|---|
| A | 0.978--1.467 | 0.0676 | 0.000--0.140 | 0.0923 |
| B | 1.957--2.935 | 0.0676 | 1.070--1.760 | 0.0885 |
| C, a=0.05 | 2.935--4.402 | 0.0676 | 1.640--2.920 | 0.0903 |
| C, a=0.15 | 3.669--5.870 | 0.0677 | 2.490--3.880 | 0.0912 |
| C, a=0.30 | 3.913--6.115 | 0.0678 | 3.190--4.620 | 0.0913 |

The maximum 5D mean polar-angle error over the prescribed window is
1.85543e-4 radians. Wider initial perturbations have longer fast
transients, while the post-fast rates stay close to the predictions.
This does not imply identical trajectories or transient times.

The initial states remain around the original common direction, and both
5D blocks have nonzero weight. These tests do not establish a global
attraction basin or extend the theorem to arbitrary sphere-wide data.

## Files and Figures

- `data/d3_results.csv` and `data/d5_results.csv` contain 153 and 68
  individual runs, including fit quality, fit counts, threshold status,
  sphere error, initial spreads, and post-fast errors.
- `data/group_summary.csv/json` summarizes the groups.
- `data/metadata.json` gives the full configuration and individual records.
- `data/baseline_regression.json` compares the 13 A cases with Figures 3(b)
  and 6 in this repository.
- `data/diagnostics/*.npz` retains all saved diagnostic arrays, initial
  states, frequencies, fit masks, and row metadata. Full evolving particle
  states are omitted. They can be regenerated locally under `_work/cache/`.
- `figures/fig_fast_transient.pdf` shows representative 3D (K=10,
  sigma_omega=0.20) and 5D (K=12) relaxation. Group C curves are medians
  over seeds; shading is the minimum--maximum range, not a confidence band.
- `figures/fig_slow_rate_sensitivity.pdf` shows individual rate ratios
  and group medians. Horizontal jitter separates overlapping points only.

The distributed validator refits the stored slow diagnostics and checks
initial-state geometry, fixed frequencies/time grids, paired random states,
threshold maintenance, and agreement with the 13 manuscript baselines.
It does not revalidate all full-particle sphere norms from compact files;
the CSV norm errors were measured in the original integrations.

## Commands

Run from the repository root.

```bash
conda run -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/run_initial_conditions.py
conda run -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/plot_initial_conditions.py
conda run -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/run_initial_conditions.py --stage self-check
```

Full recomputation uses a separate output directory and performs the
13 manuscript baseline cases before the wider initial-condition cases.

```bash
conda run --no-capture-output -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/run_initial_conditions.py --recompute --output-dir /tmp/lohe-initial-condition-recompute
```

For a smaller new-integration check, add `--stage sanity`. This selects
six representative A/B/C cases across the two dimensions. For the 13
baseline integrations only, add `--stage baselines`.
Partial runs are not reported as a completed 221-case sweep.
Omit `--output-dir` only to intentionally replace the distributed data
for this additional experiment.

