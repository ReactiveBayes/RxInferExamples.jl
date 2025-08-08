# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:27:37

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 | dim_7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rmse | 0.105493 | 0.144196 | 0.358247 | 0.157587 | 0.031505 | 0.004564 | 0.000444 |
| coverage95 | 0.974 | 0.952 | 0.947 | 1.000 | 0.962 | 0.909 | 0.849 |

## Free energy over iterations
- iterations: 2000
- start: 23.946058, end: 729166611.541615, delta: 729166587.595557

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_7/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 17380 |
| `gc_dashboard.png` | file | 184503 |
| `gc_derivative_consistency.png` | file | 41934 |
| `gc_errors.png` | file | 110748 |
| `gc_fe_cumsum.png` | file | 25426 |
| `gc_fe_iterations.png` | file | 28066 |
| `gc_free_energy_timeseries.csv` | file | 135836 |
| `gc_mse_time.png` | file | 226557 |
| `gc_residual_hist.png` | file | 12237 |
| `gc_residuals.png` | file | 70890 |
| `gc_rmse.png` | file | 14931 |
| `gc_scatter_true_vs_inferred.png` | file | 153009 |
| `gc_state_coverage_time.png` | file | 134629 |
| `gc_states.png` | file | 158653 |
| `gc_stdres_acf.png` | file | 14584 |
| `gc_stdres_hist.png` | file | 12785 |
| `gc_stdres_qq.png` | file | 27214 |
| `gc_stdres_time.png` | file | 96101 |
| `gc_y_fit.png` | file | 25494 |
| `metrics.csv` | file | 396 |
| `post_mean.csv` | file | 287032 |
| `post_var.csv` | file | 303716 |
| `rxinfer_free_energy.csv` | file | 2014 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 287456 |
| `y.csv` | file | 46977 |
