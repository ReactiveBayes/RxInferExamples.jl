# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:29:50

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 | dim_7 | dim_8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rmse | 0.398467 | 1.128628 | 1.923330 | 1.285875 | 0.355344 | 0.059354 | 0.006094 | 0.000374 |
| coverage95 | 0.949 | 0.934 | 0.933 | 0.955 | 0.979 | 0.965 | 0.976 | 0.924 |

## Free energy over iterations
- iterations: 2000
- start: 26.522952, end: 1235760429.249666, delta: 1235760402.726714

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_8/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 17869 |
| `gc_dashboard.png` | file | 177540 |
| `gc_derivative_consistency.png` | file | 43256 |
| `gc_errors.png` | file | 30929 |
| `gc_fe_cumsum.png` | file | 19794 |
| `gc_fe_iterations.png` | file | 21137 |
| `gc_free_energy_timeseries.csv` | file | 134707 |
| `gc_mse_time.png` | file | 124577 |
| `gc_residual_hist.png` | file | 9226 |
| `gc_residuals.png` | file | 40317 |
| `gc_rmse.png` | file | 15166 |
| `gc_scatter_true_vs_inferred.png` | file | 163780 |
| `gc_state_coverage_time.png` | file | 142691 |
| `gc_states.png` | file | 150764 |
| `gc_stdres_acf.png` | file | 11534 |
| `gc_stdres_hist.png` | file | 10615 |
| `gc_stdres_qq.png` | file | 21536 |
| `gc_stdres_time.png` | file | 49887 |
| `gc_y_fit.png` | file | 24501 |
| `metrics.csv` | file | 432 |
| `post_mean.csv` | file | 321062 |
| `post_var.csv` | file | 344418 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 321169 |
| `y.csv` | file | 48433 |
