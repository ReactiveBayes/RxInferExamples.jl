### Run Flow

`run.jl` (scripted driver) does the following:
1. Creates `output/<timestamp>/`.
2. Static flow:
   - Generates `n` observations with `Environment`.
   - Builds stream and runs `infer` with `kalman_filter`, `filter_constraints`, autoupdates, initialization.
   - Animates posterior mean/variance and saves `static_inference.gif`.
   - Saves `static_free_energy.png`.
3. Realtime flow:
   - Builds a timer-driven stream at `interval_ms` cadence for `n` points.
   - Starts the engine and subscribes to a live FE stream if available.
   - Computes strict per-step free energy online every `rt_fe_every` observations (default 1). If no live FE stream is exposed, this series is persisted as realtime FE.
   - Saves matched-length artifacts: `realtime_inference.png/gif`, `realtime_free_energy.csv/png/gif`, composed estimates+FE GIF, posteriors, τ plots, stepwise metrics, and `realtime_summary.txt`.

Notes:
- It is an end-to-end executable optimized for CLI usage and artifact generation under `output/<ts>/`.
- It now writes `provenance.json`, `timings.json`, and a consolidated `summary.json` with MAE/MSE and FE stats.
- It is intentionally a thin orchestration layer and can be refactored to call the composable API in `runner.jl`.

This matches the notebook’s two sections (“static dataset” and “realtime dataset”).

`runner.jl` (composable API) provides:
- `run_static(cfg) :: RunArtifacts`: runs static inference, saves artifacts, and returns a struct with `mu`, `var`, `truth`, `obs`, `fe`, `outdir`, `n`.
- `run_realtime(cfg) :: RunArtifacts`: runs realtime inference with strict per-step FE logic, saves artifacts, and returns `RunArtifacts`.
- `compare_runs(static, realtime; outdir)`: computes metrics, writes comparison plots and `summary.json` to `outdir`.

Use `run.jl` when you want a single command that produces the full folder structure and comparison. Use `runner.jl` when you want to script experiments or integrate with other Julia code (e.g., in `sweep.jl`).

