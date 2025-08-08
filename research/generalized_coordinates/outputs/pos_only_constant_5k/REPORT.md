# GC Scenario Report: pos_only_constant_5k
Generated: 2025-08-07 20:28:20

## Scenario
- **n**: 5000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.115510 | 0.150125 | 0.359591 |
| coverage95 | 0.958 | 0.951 | 0.944 |

## Free energy over iterations
- iterations: 5000
- start: 10.814304, end: 408117980.716327, delta: 408117969.902023

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_constant_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12379 |
| `gc_dashboard.png` | file | 156615 |
| `gc_derivative_consistency.png` | file | 45867 |
| `gc_errors.png` | file | 56426 |
| `gc_fe_cumsum.png` | file | 22329 |
| `gc_fe_iterations.png` | file | 24407 |
| `gc_free_energy_timeseries.csv` | file | 341267 |
| `gc_mse_time.png` | file | 132028 |
| `gc_residual_hist.png` | file | 9315 |
| `gc_residuals.png` | file | 55457 |
| `gc_rmse.png` | file | 10253 |
| `gc_scatter_true_vs_inferred.png` | file | 65467 |
| `gc_state_coverage_time.png` | file | 92663 |
| `gc_states.png` | file | 91626 |
| `gc_stdres_acf.png` | file | 14534 |
| `gc_stdres_hist.png` | file | 10742 |
| `gc_stdres_qq.png` | file | 26841 |
| `gc_stdres_time.png` | file | 81422 |
| `gc_y_fit.png` | file | 25849 |
| `metrics.csv` | file | 116 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 134 |
