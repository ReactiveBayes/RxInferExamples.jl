### Generalized Coordinates: Constant-Acceleration Car (1D, n-order)

Demonstrates variational inference in generalized coordinates for a continuous state-space model using RxInfer.

- **State**: `x ∈ R^K` generalized coordinates up to order `K` (default `K=6`). For `K≥3`, the first three entries correspond to `[position, velocity, acceleration]`; higher orders are higher derivatives.
- **Dynamics**: constant-acceleration with Gaussian process noise.
- **Observations**: noisy position; optionally velocity.

### Contents
- `src/GeneralizedCoordinatesExamples.jl`: module entrypoint exporting `GCUtils`, `GCModel`, `GCViz`, `GCConfig`, `GCGenerators`, `GCReport`
- `src/GCUtils.jl`: n-order `(A,B,Q)` via Taylor-integrator, data generation, per-time Gaussian free-energy terms
- `src/GCModel.jl`: RxInfer `@model` and mean-field constraints (dimension-agnostic)
- `src/GCViz.jl`: plotting helpers and dashboards
- `src/GCConfig.jl`: `ScenarioConfig`, `RunConfig`, `default_run_config`
- `src/GCGenerators.jl`: truth/observation generators for multiple scenarios (constant-accel, sinusoids, polynomials, piecewise)
- `src/GCReport.jl`: minimal Markdown report writer for suite runs
- `run_gc_car.jl`: single end-to-end run script
- `run_gc_suite.jl`: batch runner across orders and scenarios
- `run_meta_analysis.jl`: aggregate metrics and plot heatmaps into `outputs/meta_analysis/`
- `test/runtests.jl`: tests
- Background: `research/generalized_coordinates/gen_coord_fep_research.md`

### Quickstart
```julia
julia --project=research/generalized_coordinates_n_order research/generalized_coordinates_n_order/run_gc_car.jl
```

Outputs are written to `research/generalized_coordinates_n_order/outputs/`.

### Installation
- Julia ≥ 1.9
- The project activates its own environment (`Project.toml`) and resolves dependencies automatically.

### Configuration knobs (edit `run_gc_car.jl`)
- `n`, `dt`: trajectory length and time-step
- `order`: number of generalized-coordinate orders (default 6)
- `σ_a`: process noise std for acceleration
- `σ_obs_pos`: observation noise std for position
- `σ_obs_vel`: set to `NaN` for position-only; set to a finite std to also observe velocity
- `x0_mean`, `x0_cov`: prior over initial state

The model matrices are built via `GCUtils.constant_acceleration_ABQ(dt; order=..., σ_a=...)` and the observation matrix `B` is selected based on whether velocity is observed.

### What the scripts do
1. Simulates ground-truth `x_true` and observations `y` in generalized coordinates.
2. Builds a linear-Gaussian SSM `@model` in `GCModel.gc_car_model`.
3. Runs RxInfer with mean-field constraints, history collection, and `free_energy=true`.
4. Saves posteriors, diagnostic plots, animations, and CSV reports.

To batch-generate outputs for multiple orders `K=1..8`, run:
```julia
julia --project=research/generalized_coordinates_n_order research/generalized_coordinates_n_order/run_gc_suite.jl
```
This produces subfolders `outputs/order_1`, ..., `outputs/order_8`, each with scenario subfolders. Then run:
```julia
julia --project=research/generalized_coordinates_n_order research/generalized_coordinates_n_order/run_meta_analysis.jl
```
to write summary heatmaps into `outputs/meta_analysis/` (RMSE, coverage, correlation, FE).

### Key outputs (in `outputs/`)
- `gc_pos.png` or `gc_pos_vel.png`: inference vs truth
- `gc_states.png`: all states with credible ribbons
- `gc_free_energy_terms.png`: per-time observation/dynamics and total terms (Gaussian approximation)
- `gc_dashboard.png`: compact overview
- `gc_y_fit.png`, `gc_stdres_hist.png`, `gc_stdres_qq.png`, `gc_stdres_acf.png`, `gc_stdres_time.png`: posterior predictive checks
- `gc_rmse.png`, `gc_mse_time.png`, `gc_coverage.png`, `gc_state_coverage_time.png`, `gc_derivative_consistency.png`: error/coverage and generalized-derivative diagnostics
- `gc_posterior_summary.csv`: posterior mean/variance by time
- `gc_free_energy_timeseries.csv`: approximate per-time FE decomposition (obs, prior, dyn, total)
- `gc_free_energy_obs_dim_terms.csv`: per-dimension observation contributions (if `R` diagonal)
- `gc_fe_iterations.png`, `rxinfer_free_energy.csv`: FE diagnostics (per-iteration; single run includes deltas)
- `gc_position_animation.gif`, `gc_states_animation.gif`: animations (when supported)

- Batch runs additionally write per-scenario: `metrics.csv`, `x_true.csv`, `post_mean.csv`, `post_var.csv`, `y.csv`, and a `REPORT.md`.

### Switching observation modalities
- Position-only: set `σ_obs_vel = NaN` (default). `B` is `1×K` with `B[1,1]=1`.
- Position+velocity: set e.g. `σ_obs_vel = 0.6`. `B` is `2×K` with `B[1,1]=1`, `B[2,2]=1`.

### Module overview
- `GCUtils`
  - `constant_acceleration_ABQ(dt; order=6, σ_a, ...)`: returns `(A, B=I_K, Q)` (K×K) with PD safeguards
  - `generate_gc_car_data(rng, n, dt; order=6, ...)`: simulates `(x, y)` sequences
  - `free_energy_timeseries(...)`: Gaussian approximation to node-level FE contributions (obs/dyn/prior)
- `GCModel`
  - `gc_car_model(y, A, B, Q, R, x0_mean, x0_cov)`: linear-Gaussian SSM in generalized coordinates
  - `make_constraints()`: mean-field `q(x,y) = q(x) q(y)`
- `GCViz`
  - Plotting and PPC utilities: state ribbons, residuals, standardized residuals, ACF/QQ, coverage, RMSE, animations
- `GCConfig`
  - `ScenarioConfig`, `RunConfig`, `default_run_config()`
- `GCGenerators`
  - Scenario truth/observation generators: `:constant_accel`, `:sinusoid`, `:sinusoid_mixed`, `:poly`, `:trend_plus_osc`, `:poly_sin_mixed`, `:piecewise_mixed`
- `GCReport`
  - `write_markdown_report(outdir, scen; extra=Dict())`

### Tests
```julia
julia --project=research/generalized_coordinates_n_order -e 'using Pkg; Pkg.test()'
```
Tests cover:
- Inference fidelity under position-only and pos+vel observations
- Free-energy reporting and monotonicity sanity checks
- Run-script smoke test and presence of output CSVs (`rxinfer_free_energy.csv`, `gc_posterior_summary.csv`, `gc_free_energy_timeseries.csv`)

### Tips and troubleshooting
- If plots are blank in headless environments, the backend falls back to a static PNG setup (`GCViz` sets GR non-interactive).
- If `Pkg` spends time precompiling, set `ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"` as in `run_gc_car.jl`.
- For longer GIFs, ensure the environment supports writing large animations or disable animation blocks.

### Technical background
See `research/generalized_coordinates/gen_coord_fep_research.md` and the accompanying `TECHNICAL_README.md` for the full mathematical treatment (FEP, generalized filtering, HGF, variational message passing) and how they map onto the implementation here.
