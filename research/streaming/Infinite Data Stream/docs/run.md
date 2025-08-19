### Run Flow

`run.jl` does the following:
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

This matches the notebook’s two sections (“static dataset” and “realtime dataset”).

