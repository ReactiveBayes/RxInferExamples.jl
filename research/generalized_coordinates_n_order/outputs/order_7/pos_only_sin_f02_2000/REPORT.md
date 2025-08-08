# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 09:40:27

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 | dim_7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rmse | 0.281557 | 0.419695 | 0.952242 | 0.706979 | 0.140553 | 0.017478 | 0.001339 |
| coverage95 | 0.173 | 0.180 | 0.355 | 0.603 | 0.441 | 0.541 | 0.441 |

## Free energy over iterations
- iterations: 2000
- start: 23.568556, end: 227462953.214624, delta: 227462929.646067

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_7/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 17182 |
| `gc_dashboard.png` | file | 295168 |
| `gc_derivative_consistency.png` | file | 35725 |
| `gc_errors.png` | file | 47138 |
| `gc_fe_cumsum.png` | file | 22281 |
| `gc_fe_iterations.png` | file | 32694 |
| `gc_free_energy_timeseries.csv` | file | 138492 |
| `gc_mse_time.png` | file | 165983 |
| `gc_residual_hist.png` | file | 11679 |
| `gc_residuals.png` | file | 66942 |
| `gc_rmse.png` | file | 15799 |
| `gc_scatter_true_vs_inferred.png` | file | 121933 |
| `gc_state_coverage_time.png` | file | 147303 |
| `gc_states.png` | file | 271121 |
| `gc_stdres_acf.png` | file | 13134 |
| `gc_stdres_hist.png` | file | 12758 |
| `gc_stdres_qq.png` | file | 28303 |
| `gc_stdres_time.png` | file | 85588 |
| `gc_y_fit.png` | file | 143739 |
| `metrics.csv` | file | 248 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
