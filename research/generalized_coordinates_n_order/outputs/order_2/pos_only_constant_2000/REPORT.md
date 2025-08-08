# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:21:16

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 |
|---|---:|---:|
| rmse | 0.019279 | 0.000420 |
| coverage95 | 1.000 | 0.934 |

## Free energy over iterations
- iterations: 2000
- start: 6.806018, end: 889094.392823, delta: 889087.586805

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_2/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12764 |
| `gc_dashboard.png` | file | 90865 |
| `gc_derivative_consistency.png` | file | 19585 |
| `gc_errors.png` | file | 27441 |
| `gc_fe_cumsum.png` | file | 21267 |
| `gc_fe_iterations.png` | file | 32328 |
| `gc_free_energy_timeseries.csv` | file | 129024 |
| `gc_mse_time.png` | file | 62061 |
| `gc_residual_hist.png` | file | 9621 |
| `gc_residuals.png` | file | 69739 |
| `gc_rmse.png` | file | 11868 |
| `gc_scatter_true_vs_inferred.png` | file | 49229 |
| `gc_state_coverage_time.png` | file | 31222 |
| `gc_states.png` | file | 65412 |
| `gc_stdres_acf.png` | file | 13671 |
| `gc_stdres_hist.png` | file | 10115 |
| `gc_stdres_qq.png` | file | 24987 |
| `gc_stdres_time.png` | file | 93253 |
| `gc_y_fit.png` | file | 30052 |
| `metrics.csv` | file | 136 |
| `post_mean.csv` | file | 83548 |
| `post_var.csv` | file | 96788 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 83452 |
| `y.csv` | file | 45783 |
