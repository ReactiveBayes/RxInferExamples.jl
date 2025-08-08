# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 09:42:52

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 | dim_7 | dim_8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rmse | 0.293498 | 0.471068 | 1.128642 | 1.081439 | 0.368151 | 0.081775 | 0.012043 | 0.001107 |
| coverage95 | 0.166 | 0.170 | 0.350 | 0.731 | 0.548 | 0.514 | 0.451 | 0.531 |

## Free energy over iterations
- iterations: 2000
- start: 32.014110, end: 429047258.090900, delta: 429047226.076790

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_8/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 17561 |
| `gc_dashboard.png` | file | 300550 |
| `gc_derivative_consistency.png` | file | 40648 |
| `gc_errors.png` | file | 46637 |
| `gc_fe_cumsum.png` | file | 24935 |
| `gc_fe_iterations.png` | file | 32962 |
| `gc_free_energy_timeseries.csv` | file | 138630 |
| `gc_mse_time.png` | file | 160770 |
| `gc_residual_hist.png` | file | 12049 |
| `gc_residuals.png` | file | 64789 |
| `gc_rmse.png` | file | 16279 |
| `gc_scatter_true_vs_inferred.png` | file | 137663 |
| `gc_state_coverage_time.png` | file | 151519 |
| `gc_states.png` | file | 278886 |
| `gc_stdres_acf.png` | file | 12892 |
| `gc_stdres_hist.png` | file | 12885 |
| `gc_stdres_qq.png` | file | 27109 |
| `gc_stdres_time.png` | file | 79642 |
| `gc_y_fit.png` | file | 141220 |
| `metrics.csv` | file | 282 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
