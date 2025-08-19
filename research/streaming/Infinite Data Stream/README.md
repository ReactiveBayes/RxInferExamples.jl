### Infinite Data Stream (modular)

Entry points:
- `run_static.jl`: runs inference on a pre-generated static stream and saves `infinite-data-stream-inference.gif`.
- `run_realtime.jl`: runs inference on a timer-based stream (41ms cadence).

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

