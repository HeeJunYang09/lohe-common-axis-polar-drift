# Additional Numerical Experiments

These experiments supplement the manuscript figures with multi-target
hitting-time validation and initial-condition sensitivity tests. They are
additional numerical evidence, not numbered figures in the manuscript.

| Experiment | Purpose | Connection to the manuscript |
|---|---|---|
| [Multi-target hitting time](multi_target_hitting_time/README.md) | Test the logarithmic dependence on the target polar angle | Figure 4, with the same 25 deterministic parameter pairs |
| [Initial-condition sensitivity](initial_condition_sensitivity/README.md) | Change initialization while holding the numerical settings fixed | Figure 3(b) in three dimensions and Figure 6 in five dimensions |

## Reproduction

Run the following commands from the repository root in the environment
described by the root `requirements-lock.txt`. No additional packages are
required. The commands read the included compact data and do not integrate
new trajectories.

```bash
conda run -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/run_multi_target.py
conda run -n py3_9 python -B additional_experiments/multi_target_hitting_time/scripts/plot_multi_target.py
conda run -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/run_initial_conditions.py
conda run -n py3_9 python -B additional_experiments/initial_condition_sensitivity/scripts/plot_initial_conditions.py
conda run -n py3_9 python -B -m pytest -q -p no:cacheprovider additional_experiments/tests
```

The run scripts validate stored numerical records against the manuscript
caches in this repository. Plotting uses the shared `src/figure_style.py`.
Each experiment writes only inside its own directory by default. Figures
are produced as PDF and PNG; PDF files are the public figure artifacts.

## Data and Full Recomputation

The included data were exported from completed integrations without changing
numerical values. They retain individual-run diagnostics, initial states
where applicable, parameter settings, and regression comparisons. Large
full-particle trajectories are deliberately not distributed.

Full integration requires an explicit `--recompute` flag. Each experiment
README gives its command and numerical protocol. Intermediate full-state
caches are stored in the ignored `_work/` directory. Use a fresh
`--output-dir` for a separate recomputation without replacing distributed
results. The original manuscript `src/`, `scripts/`, `data/`, and
`figures/` are read-only inputs to these additional experiments.

Stored validation summaries describe the original calculations. Running the
validation commands checks the distributed diagnostics again; it does not
constitute a new full integration or a new validation of every particle state.
Neither experiment establishes a global attraction basin or a uniform
asymptotic error bound at the pole.

