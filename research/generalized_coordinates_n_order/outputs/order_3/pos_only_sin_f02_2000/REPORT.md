# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 09:33:22

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 |
|---|---:|---:|---:|
| rmse | 0.299687 | 0.398839 | 0.741666 |
| coverage95 | 0.164 | 0.168 | 0.351 |

## Free energy over iterations
- iterations: 2000
- start: 10.394814, end: 100051484.211921, delta: 100051473.817107

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_3/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 14265 |
| `gc_dashboard.png` | file | 221750 |
| `gc_derivative_consistency.png` | file | 33436 |
| `gc_errors.png` | file | 48227 |
| `gc_fe_cumsum.png` | file | 21563 |
| `gc_fe_iterations.png` | file | 27218 |
| `gc_free_energy_timeseries.csv` | file | 136201 |
| `gc_mse_time.png` | file | 99847 |
| `gc_residual_hist.png` | file | 11137 |
| `gc_residuals.png` | file | 69260 |
| `gc_rmse.png` | file | 12261 |
| `gc_scatter_true_vs_inferred.png` | file | 74461 |
| `gc_state_coverage_time.png` | file | 96426 |
| `gc_states.png` | file | 197491 |
| `gc_stdres_acf.png` | file | 12678 |
| `gc_stdres_hist.png` | file | 12998 |
| `gc_stdres_qq.png` | file | 28935 |
| `gc_stdres_time.png` | file | 90439 |
| `gc_y_fit.png` | file | 143255 |
| `metrics.csv` | file | 117 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
