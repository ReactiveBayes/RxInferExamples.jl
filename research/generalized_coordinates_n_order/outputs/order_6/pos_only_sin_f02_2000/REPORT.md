# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 10:26:10

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 |
|---|---:|---:|---:|---:|---:|---:|
| rmse | 0.273510 | 0.385062 | 0.814111 | 0.406103 | 0.032735 | 0.001485 |
| coverage95 | 0.179 | 0.186 | 0.374 | 0.368 | 0.251 | 0.198 |

## Free energy over iterations
- iterations: 2000
- start: 20.101637, end: 140533638.876992, delta: 140533618.775355

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_6/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 16395 |
| `gc_dashboard.png` | file | 291847 |
| `gc_derivative_consistency.png` | file | 44778 |
| `gc_errors.png` | file | 47560 |
| `gc_fe_cumsum.png` | file | 21366 |
| `gc_fe_iterations.png` | file | 32603 |
| `gc_free_energy_timeseries.csv` | file | 139240 |
| `gc_mse_time.png` | file | 153019 |
| `gc_residual_hist.png` | file | 11722 |
| `gc_residuals.png` | file | 65936 |
| `gc_rmse.png` | file | 15060 |
| `gc_scatter_true_vs_inferred.png` | file | 120031 |
| `gc_state_coverage_time.png` | file | 133275 |
| `gc_states.png` | file | 266707 |
| `gc_stdres_acf.png` | file | 12893 |
| `gc_stdres_hist.png` | file | 12914 |
| `gc_stdres_qq.png` | file | 28046 |
| `gc_stdres_time.png` | file | 81639 |
| `gc_y_fit.png` | file | 142905 |
| `metrics.csv` | file | 293 |
| `post_mean.csv` | file | 248348 |
| `post_var.csv` | file | 262936 |
| `rxinfer_free_energy.csv` | file | 2014 |
| `scenario_config.toml` | file | 139 |
| `x_true.csv` | file | 145636 |
| `y.csv` | file | 46789 |
