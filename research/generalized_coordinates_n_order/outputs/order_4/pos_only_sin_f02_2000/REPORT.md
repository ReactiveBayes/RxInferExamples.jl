# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 09:34:02

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 |
|---|---:|---:|---:|---:|
| rmse | 0.297800 | 0.403143 | 0.776756 | 0.155823 |
| coverage95 | 0.169 | 0.171 | 0.346 | 0.000 |

## Free energy over iterations
- iterations: 2000
- start: 13.396525, end: 102727612.210951, delta: 102727598.814427

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_4/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 15105 |
| `gc_dashboard.png` | file | 229922 |
| `gc_derivative_consistency.png` | file | 39638 |
| `gc_errors.png` | file | 47573 |
| `gc_fe_cumsum.png` | file | 23081 |
| `gc_fe_iterations.png` | file | 27340 |
| `gc_free_energy_timeseries.csv` | file | 135706 |
| `gc_mse_time.png` | file | 127310 |
| `gc_residual_hist.png` | file | 11633 |
| `gc_residuals.png` | file | 67215 |
| `gc_rmse.png` | file | 13116 |
| `gc_scatter_true_vs_inferred.png` | file | 86383 |
| `gc_state_coverage_time.png` | file | 108941 |
| `gc_states.png` | file | 214779 |
| `gc_stdres_acf.png` | file | 12837 |
| `gc_stdres_hist.png` | file | 13009 |
| `gc_stdres_qq.png` | file | 28144 |
| `gc_stdres_time.png` | file | 84805 |
| `gc_y_fit.png` | file | 141391 |
| `metrics.csv` | file | 148 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
