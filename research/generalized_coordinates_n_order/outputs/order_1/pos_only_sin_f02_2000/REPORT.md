# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 09:32:15

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 |
|---|---:|
| rmse | 3.535542 |
| coverage95 | 0.000 |

## Free energy over iterations
- iterations: 2000
- start: 3.027913, end: 45654.885743, delta: 45651.857830

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_1/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 11412 |
| `gc_dashboard.png` | file | 63095 |
| `gc_derivative_consistency.png` | file | 15338 |
| `gc_errors.png` | file | 50386 |
| `gc_fe_cumsum.png` | file | 21821 |
| `gc_fe_iterations.png` | file | 24728 |
| `gc_free_energy_timeseries.csv` | file | 126653 |
| `gc_mse_time.png` | file | 52045 |
| `gc_residual_hist.png` | file | 11126 |
| `gc_residuals.png` | file | 58787 |
| `gc_rmse.png` | file | 8459 |
| `gc_scatter_true_vs_inferred.png` | file | 17557 |
| `gc_state_coverage_time.png` | file | 13156 |
| `gc_states.png` | file | 35624 |
| `gc_stdres_acf.png` | file | 14598 |
| `gc_stdres_hist.png` | file | 13433 |
| `gc_stdres_qq.png` | file | 25489 |
| `gc_stdres_time.png` | file | 60270 |
| `gc_y_fit.png` | file | 131532 |
| `metrics.csv` | file | 50 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
