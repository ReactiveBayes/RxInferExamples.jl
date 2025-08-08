# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:20:52

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 |
|---|---:|
| rmse | 0.010205 |
| coverage95 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 3.729559, end: 125656.781001, delta: 125653.051442

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_1/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 11548 |
| `gc_dashboard.png` | file | 42437 |
| `gc_derivative_consistency.png` | file | 15338 |
| `gc_errors.png` | file | 37365 |
| `gc_fe_cumsum.png` | file | 24275 |
| `gc_fe_iterations.png` | file | 24975 |
| `gc_free_energy_timeseries.csv` | file | 130547 |
| `gc_mse_time.png` | file | 38253 |
| `gc_residual_hist.png` | file | 9723 |
| `gc_residuals.png` | file | 70275 |
| `gc_rmse.png` | file | 12535 |
| `gc_scatter_true_vs_inferred.png` | file | 21427 |
| `gc_state_coverage_time.png` | file | 13170 |
| `gc_states.png` | file | 19575 |
| `gc_stdres_acf.png` | file | 13795 |
| `gc_stdres_hist.png` | file | 10770 |
| `gc_stdres_qq.png` | file | 25372 |
| `gc_stdres_time.png` | file | 94873 |
| `gc_y_fit.png` | file | 153207 |
| `metrics.csv` | file | 82 |
| `post_mean.csv` | file | 50474 |
| `post_var.csv` | file | 54255 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 53412 |
| `y.csv` | file | 48699 |
