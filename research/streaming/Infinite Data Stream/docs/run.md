### Run Flow

`run.jl` does the following:
1. Creates `output/<timestamp>/`.
2. Static flow:
   - Generates `n` observations with `Environment`.
   - Builds stream and runs `infer` with `kalman_filter`, `filter_constraints`, autoupdates, initialization.
   - Animates posterior mean/variance and saves `static_inference.gif`.
   - Saves `static_free_energy.png`.
3. Realtime flow:
   - Builds a timer-driven stream at ~41ms cadence for `n` points.
   - Starts the engine; after expected duration, writes `realtime_summary.txt`.

This matches the notebook’s two sections (“static dataset” and “realtime dataset”).

