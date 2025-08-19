### Infinite Data Stream (modular)

Entry points:
- `run_static.jl`: runs inference on a pre-generated static stream (headless PNG by default, opt-in GIF).
- `run_realtime.jl`: runs inference on a timer-based stream (41ms cadence).
- `run.jl`: unified driver that writes to `output/<timestamp>/{static,realtime,comparison}`

Structure:
- `model.jl`: RxInfer `@model` and `@constraints`.
- `environment.jl`: signal generator and observation utilities.
- `updates.jl`: `@autoupdates` and initialization builders.
- `streams.jl`: stream helpers (Rocket pipelines).
- `visualize.jl`: plotting helpers.
- `utils.jl`: helper to `include` modules when running scripts directly.
- `config.toml`: central configuration (n, intervals, iterations, GIF settings, output dir).

Usage (Julia):
```julia
include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
include("run_static.jl")
```

Headless full run (configurable via `config.toml`, env can override):
```bash
GKSwstype=100 \
IDS_MAKE_GIF=1 IDS_GIF_STRIDE=5 IDS_RT_GIF_STRIDE=6 \
IDS_RT_ITERATIONS=8 IDS_RT_FE_EVERY=1 \
julia --project=. run.jl
```

Visual tools exported in `visualize.jl`:
- `plot_hidden_and_obs`, `plot_estimates`, `animate_estimates`
- `plot_tau`, `plot_overlay_means`, `plot_scatter_static_vs_realtime`, `plot_residuals`
 - `animate_free_energy`, `animate_composed_estimates_fe`, `animate_overlay_means`,
 - `plot_fe_comparison`, `animate_comparison_static_vs_realtime`, `plot_tau_comparison`

Advanced:
- Swap `Environment` with a custom producer for heterogeneous empirical data
- Control GIF stride via `IDS_GIF_STRIDE` or `IDS_RT_GIF_STRIDE`
 - Toggle animation generation with `IDS_MAKE_GIF=1` (writes `static_free_energy.gif`, `static_composed_estimates_fe.gif`, `realtime_inference.gif`, `realtime_free_energy.gif`, and `comparison/overlay_means.gif`)
 - Strict stepwise FE minimization online in realtime is enabled by default. Control frequency with `rt_fe_every` in `config.toml` or `IDS_RT_FE_EVERY` (1 = every point). If the engine exposes a live FE stream, that is used; otherwise a strict per-prefix FE is computed online; as a last resort, an offline per-prefix FE is computed after the run.

Artifacts parity (static vs realtime):
- Static writes: `static_inference.png/gif`, `static_free_energy.csv/png/gif`, `static_composed_estimates_fe.gif`, `static_posterior_x_current.csv`, `static_posterior_tau_shape_rate.csv`, `static_tau_mean.png`, `static_truth_history.csv`, `static_observations.csv`, `static_metrics_stepwise.csv`.
- Realtime writes: `realtime_inference.png/gif`, `realtime_free_energy.csv/png/gif`, `realtime_composed_estimates_fe.gif`, `realtime_posterior_x_current.csv`, `realtime_posterior_tau_shape_rate.csv`, `realtime_tau_mean.png`, `realtime_truth_history.csv`, `realtime_observations.csv`, `realtime_metrics_stepwise.csv`, `realtime_summary.txt`.

Configuration keys (config.toml | ENV override):
- `n` | `IDS_N`: number of timesteps
- `interval_ms` | `IDS_INTERVAL_MS`: realtime cadence
- `iterations` | `IDS_ITERATIONS`: static per-step iterations
- `rt_iterations` | `IDS_RT_ITERATIONS`: realtime per-step iterations
- `rt_fe_every` | `IDS_RT_FE_EVERY`: compute FE every k observations online (1 for every point)
- `gif_stride` | `IDS_GIF_STRIDE`, `rt_gif_stride` | `IDS_RT_GIF_STRIDE`, `make_gif` | `IDS_MAKE_GIF`
- `seed` | `IDS_SEED`, `keephistory`, `output_dir`

Provenance (`output/<ts>/provenance.json`) includes: `n`, `interval_ms`, `iterations`, `rt_iterations`, `keephistory`, `rt_fe_every`, `seed`, and GIF/paths.

