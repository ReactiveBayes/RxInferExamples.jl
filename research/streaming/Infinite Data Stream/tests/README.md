### Tests overview

This test suite validates the modular Infinite Data Stream refactor across environment, streams, model/updates, visualization, utilities, configuration, and orchestration runners.

What is covered:
- `unit_environment.jl`: synthetic signal generator (`Environment`, `getnext!`, `gethistory`, `getobservations`).
- `unit_streams.jl`: `to_namedtuple_stream` and `timer_observations` helpers.
- `unit_model_updates.jl`: `@autoupdates` and `@initialization` builders; smoke inference run with free energy tracking.
- `unit_visualize.jl`: plotting helpers and minimal animation smoke; can skip animations via `SKIP_ANIMATION_TESTS=1`.
- `unit_meta_project_docs.jl`: presence of `Project.toml`, `Manifest.toml`, `meta.jl`, `README.md`, and `docs/`.
- `unit_free_energy.jl`: per-timestep free energy computation on short sequences using the same model/updates.
- `runtests.jl` integration (optional): end-to-end `run.jl` pipeline and smoke runs for `run_static.jl`/`run_realtime.jl` with reduced problem sizes.

Timeouts and reliability:
- Each unit test is executed in its own Julia process with a 30s timeout; hung processes are killed and reported.
- Animations in tests are minimized; set `SKIP_ANIMATION_TESTS=1` to skip them entirely.
- CI-friendly overrides are supported via environment variables (see below).

How to run:
```bash
# From directory: research/streaming/Infinite Data Stream
julia --project=. -e 'include(joinpath(@__DIR__, "tests", "runtests.jl"))'

# Recommended CI settings (fast, deterministic):
SKIP_ANIMATION_TESTS=1 \
RUN_INTEGRATION_TESTS=0 \
IDS_N=90 IDS_INTERVAL_MS=3 IDS_ITERATIONS=3 \
julia --project=. -e 'include(joinpath(@__DIR__, "tests", "runtests.jl"))'

# To include the full integration test of run.jl (writes to output/<timestamp>):
RUN_INTEGRATION_TESTS=1 \
SKIP_ANIMATION_TESTS=1 \
julia --project=. -e 'include(joinpath(@__DIR__, "tests", "runtests.jl"))'
```

Key artifacts and methods validated:
- Config loading (`config.toml`) via `InfiniteDataStreamUtils.load_config()` and ENV overrides (`IDS_*`).
- Environment dynamics and observation generation (`environment.jl`).
- Model and constraints (`model.jl`), autoupdates and initialization (`updates.jl`).
- Stream utilities for static and realtime (`streams.jl`).
- Visualization functions, including composed estimate+FE animations (`visualize.jl`).
- Unified orchestration in `run.jl`, plus `run_static.jl` and `run_realtime.jl` entry-points.
- Output structure and comparison plots (`static_inference.png`, `means_compare.png`, residual plots, GIFs).

Expected outputs when `RUN_INTEGRATION_TESTS=1`:
- `output/<timestamp>/static/`: `static_inference.png`, `static_free_energy.csv/png/gif`, `static_composed_estimates_fe.gif`, posterior CSVs, tau plots.
- `output/<timestamp>/realtime/`: `realtime_inference.png/gif`, `realtime_free_energy.csv/gif` (mirrored if FE stream unavailable), posterior CSVs, summary.
- `output/<timestamp>/comparison/`: `metrics.txt`, `means_compare.png`, residual plots, `scatter_static_vs_realtime.png`, optional `overlay_means.gif`.

Troubleshooting:
- If tests appear slow, confirm headless plotting via `GKSwstype=100` and enable `SKIP_ANIMATION_TESTS=1`.
- Use `IDS_SKIP_FE_REPLOT=1` to avoid recomputing FE twice during `run.jl` static branch.
- Ensure project activation is correct when running from other directories.


