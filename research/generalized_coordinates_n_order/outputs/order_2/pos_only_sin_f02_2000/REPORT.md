# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 10:21:24

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 |
|---|---:|---:|
| rmse | 3.531992 | 4.442879 |
| coverage95 | 0.002 | 0.000 |

## Free energy over iterations
- iterations: 2000
- start: 7.803380, end: 293199.893118, delta: 293192.089738

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_2/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12589 |
| `gc_dashboard.png` | file | 102435 |
| `gc_derivative_consistency.png` | file | 18115 |
| `gc_errors.png` | file | 90826 |
| `gc_fe_cumsum.png` | file | 23071 |
| `gc_fe_iterations.png` | file | 31767 |
| `gc_free_energy_timeseries.csv` | file | 126637 |
| `gc_mse_time.png` | file | 102705 |
| `gc_residual_hist.png` | file | 11176 |
| `gc_residuals.png` | file | 58552 |
| `gc_rmse.png` | file | 10051 |
| `gc_scatter_true_vs_inferred.png` | file | 42627 |
| `gc_state_coverage_time.png` | file | 28169 |
| `gc_states.png` | file | 68422 |
| `gc_stdres_acf.png` | file | 14628 |
| `gc_stdres_hist.png` | file | 13274 |
| `gc_stdres_qq.png` | file | 26049 |
| `gc_stdres_time.png` | file | 60442 |
| `gc_y_fit.png` | file | 131049 |
| `metrics.csv` | file | 133 |
| `post_mean.csv` | file | 94270 |
| `post_var.csv` | file | 95073 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
| `x_true.csv` | file | 84004 |
| `y.csv` | file | 46828 |
