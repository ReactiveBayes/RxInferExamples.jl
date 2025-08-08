# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 09:33:08

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 |
|---|---:|---:|---:|
| rmse | 0.120408 | 0.150103 | 0.355001 |
| coverage95 | 0.956 | 0.938 | 0.956 |

## Free energy over iterations
- iterations: 2000
- start: 10.080071, end: 408117980.649179, delta: 408117970.569108

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_3/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 14325 |
| `gc_dashboard.png` | file | 109044 |
| `gc_derivative_consistency.png` | file | 42137 |
| `gc_errors.png` | file | 102897 |
| `gc_fe_cumsum.png` | file | 21270 |
| `gc_fe_iterations.png` | file | 24820 |
| `gc_free_energy_timeseries.csv` | file | 135821 |
| `gc_mse_time.png` | file | 166793 |
| `gc_residual_hist.png` | file | 12572 |
| `gc_residuals.png` | file | 71586 |
| `gc_rmse.png` | file | 12054 |
| `gc_scatter_true_vs_inferred.png` | file | 65485 |
| `gc_state_coverage_time.png` | file | 83916 |
| `gc_states.png` | file | 91135 |
| `gc_stdres_acf.png` | file | 14493 |
| `gc_stdres_hist.png` | file | 12921 |
| `gc_stdres_qq.png` | file | 26571 |
| `gc_stdres_time.png` | file | 95254 |
| `gc_y_fit.png` | file | 27028 |
| `metrics.csv` | file | 119 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 146 |
