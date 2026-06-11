# Reproduction Data

## `cache/`

This directory contains the small processed NumPy `.npz` cache files used to
reproduce Figures 1--5 without rerunning the numerical integrations.

- `fig01_fast_locking.npz`: Figure 1 deterministic fast-locking data
- `fig02_phase_gap_prediction.npz`: Figure 2 phase-gap and scaling data
- `fig03_slow_polar_drift_deterministic_3x3.npz`: Figure 3 deterministic data
- `fig04_hitting_time_deterministic_5x5.npz`: Figure 4 deterministic data
- `fig03_slow_polar_drift_gaussian_3x3_seeds0-4.npz`: Figure 5(a) Gaussian data
- `fig04_hitting_time_gaussian_5x5_seeds0-4.npz`: Figure 5(b) Gaussian data

## `processed/`

The figure scripts write human-readable CSV, JSON, and text diagnostic
summaries to this directory when they run.
