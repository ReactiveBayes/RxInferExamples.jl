# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:25:26

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 |
|---|---:|---:|---:|---:|---:|---:|
| rmse | 0.119585 | 0.140993 | 0.345500 | 0.194646 | 0.011276 | 0.000491 |
| coverage95 | 0.963 | 0.957 | 0.949 | 0.832 | 0.837 | 0.866 |

## Free energy over iterations
- iterations: 2000
- start: 19.930744, end: 505956747.775333, delta: 505956727.844589

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_6/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 16697 |
| `gc_dashboard.png` | file | 168369 |
| `gc_derivative_consistency.png` | file | 45058 |
| `gc_errors.png` | file | 110125 |
| `gc_fe_cumsum.png` | file | 22994 |
| `gc_fe_iterations.png` | file | 28598 |
| `gc_free_energy_timeseries.csv` | file | 135709 |
| `gc_mse_time.png` | file | 232019 |
| `gc_residual_hist.png` | file | 9383 |
| `gc_residuals.png` | file | 64690 |
| `gc_rmse.png` | file | 14159 |
| `gc_scatter_true_vs_inferred.png` | file | 126444 |
| `gc_state_coverage_time.png` | file | 122360 |
| `gc_states.png` | file | 145292 |
| `gc_stdres_acf.png` | file | 14670 |
| `gc_stdres_hist.png` | file | 10545 |
| `gc_stdres_qq.png` | file | 25071 |
| `gc_stdres_time.png` | file | 86542 |
| `gc_y_fit.png` | file | 26203 |
| `metrics.csv` | file | 342 |
| `post_mean.csv` | file | 252398 |
| `post_var.csv` | file | 258708 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 253123 |
| `y.csv` | file | 48071 |
