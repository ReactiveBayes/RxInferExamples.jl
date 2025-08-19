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
GKSwstype=100 IDS_MAKE_GIF=1 IDS_GIF_STRIDE=5 IDS_RT_GIF_STRIDE=6 \
julia --project=. run.jl
```

Visual tools exported in `visualize.jl`:
- `plot_hidden_and_obs`, `plot_estimates`, `animate_estimates`
- `plot_tau`, `plot_overlay_means`, `plot_scatter_static_vs_realtime`, `plot_residuals`

Advanced:
- Swap `Environment` with a custom producer for heterogeneous empirical data
- Control GIF stride via `IDS_GIF_STRIDE` or `IDS_RT_GIF_STRIDE`
 - Toggle animation generation with `IDS_MAKE_GIF=1` (writes `static_free_energy.gif`, `static_composed_estimates_fe.gif`, `realtime_inference.gif`, and `comparison/overlay_means.gif`)

