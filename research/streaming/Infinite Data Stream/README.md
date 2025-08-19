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

Usage (Julia):
```julia
include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
include("run_static.jl")
```

Headless full run with realtime GIF stride 6:
```bash
env GKSwstype=100 IDS_RT_GIF_STRIDE=6 \
julia --project=. run.jl
```

Visual tools exported in `visualize.jl`:
- `plot_hidden_and_obs`, `plot_estimates`, `animate_estimates`
- `plot_tau`, `plot_overlay_means`, `plot_scatter_static_vs_realtime`, `plot_residuals`

Advanced:
- Swap `Environment` with a custom producer for heterogeneous empirical data
- Control GIF stride via `IDS_GIF_STRIDE` or `IDS_RT_GIF_STRIDE`

