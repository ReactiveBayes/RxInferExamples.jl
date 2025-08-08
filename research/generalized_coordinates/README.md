### Generalized Coordinates: Constant-Acceleration Car (1D)

Demonstrates variational inference in generalized coordinates for a continuous state-space model using RxInfer.

- **State**: `x = [position, velocity, acceleration]`.
- **Dynamics**: constant-acceleration with Gaussian process noise.
- **Observations**: noisy position; optionally velocity.

Layout
- `src/GeneralizedCoordinatesExamples.jl`: module entrypoint exporting `GCUtils`, `GCModel`, `GCViz`
- `src/GCUtils.jl`: data generation, (A,B,Q) matrices, per-time Gaussian FE terms
- `src/GCModel.jl`: RxInfer `@model` and constraints
- `src/GCViz.jl`: plotting helpers and dashboards
- `run_gc_car.jl`: end-to-end run script
- `test/runtests.jl`: tests

Run
```julia
julia --project=research/generalized_coordinates research/generalized_coordinates/run_gc_car.jl
```

Outputs (saved to `research/generalized_coordinates/outputs/`)
- `gc_pos.png` or `gc_pos_vel.png`
- `gc_free_energy_terms.png`
- `gc_dashboard.png`
- `gc_free_energy_timeseries.csv`
- `gc_position_animation.gif` (if position-only observation)

Tests
```julia
julia --project=research/generalized_coordinates -e 'using Pkg; Pkg.test()'
```
