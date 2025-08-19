### Project and Entry Points

Files:
- `Project.toml` / `Manifest.toml`: self-contained environment (`RxInfer`, `Rocket`, `Plots`, `StableRNGs`).
- `README.md`: quick-start and structure.
- `run.jl`: unified CLI driver. Creates `output/<timestamp>/` and runs both static and realtime, writes `provenance.json`, `timings.json`, and a run-level `summary.json`.
- `run_static.jl` / `run_realtime.jl`: standalone entry points for focused runs.
- `config.toml`: central configuration for number of steps, timer cadence, inference iterations/history, and GIF/output settings.
  - Supports `seed` (or `IDS_SEED`) for reproducible environment generation.
  - Realtime-specific keys: `rt_iterations` (or `IDS_RT_ITERATIONS`) controls per-step updates; `rt_fe_every` (or `IDS_RT_FE_EVERY`) controls strict online FE frequency (1 = every point).
  - CLI flags in `run.jl` override `config.toml` and ENV: `--n`, `--interval_ms`, `--iterations`, `--rt_iterations`, `--rt_fe_every`, `--keephistory`, `--gif_stride`, `--rt_gif_stride`, `--seed`, `--make_gif`, `--output_dir`.
- `runner.jl`: composable programmatic API (`run_static`, `run_realtime`, `compare_runs`) returning `RunArtifacts`. Prefer this for notebooks, custom scripts, and sweeps.

How to run:
```julia
# one-shot end-to-end
julia --project=. run.jl

# or, load modules and run static only
include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
include("run_static.jl")
```

Outputs:
- `output/<ts>/static/{static_inference.png,gif,static_free_energy.csv,png,gif,static_composed_estimates_fe.gif,static_posterior_x_current.csv,static_posterior_tau_shape_rate.csv,static_tau_mean.png,static_truth_history.csv,static_observations.csv,static_metrics_stepwise.csv}`
- `output/<ts>/realtime/{realtime_inference.png,gif,realtime_free_energy.csv,png,gif,realtime_composed_estimates_fe.gif,realtime_posterior_x_current.csv,realtime_posterior_tau_shape_rate.csv,realtime_tau_mean.png,realtime_truth_history.csv,realtime_observations.csv,realtime_metrics_stepwise.csv,realtime_summary.txt}`
- `output/<ts>/comparison/{metrics.txt,means_compare.png,scatter_static_vs_realtime.png,residuals_static.png,residuals_realtime.png,overlay_means.gif,free_energy_compare.png}`

