# GC Scenario Report: pos_only_trend_osc_5k
Generated: 2025-08-07 20:32:16

## Scenario
- **n**: 5000
- **dt**: 0.05
- **σ_a**: 0.2
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `trend_plus_osc` with kwargs `Dict{Symbol, Any}(:degree => 2, :coeffs => [0.0, 0.2, 0.0], :freq => 0.15, :amp => 1.5)`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.069380 | 0.082629 | 0.156186 |
| coverage95 | 0.974 | 0.998 | 1.000 |

## Free energy over iterations
- iterations: 5000
- start: 11.144054, end: 254671019.813002, delta: 254671008.668948

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_trend_osc_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12341 |
| `gc_dashboard.png` | file | 141841 |
| `gc_derivative_consistency.png` | file | 37772 |
| `gc_errors.png` | file | 64423 |
| `gc_fe_cumsum.png` | file | 24011 |
| `gc_fe_iterations.png` | file | 27845 |
| `gc_free_energy_timeseries.csv` | file | 347359 |
| `gc_mse_time.png` | file | 108978 |
| `gc_residual_hist.png` | file | 12532 |
| `gc_residuals.png` | file | 55384 |
| `gc_rmse.png` | file | 10616 |
| `gc_scatter_true_vs_inferred.png` | file | 99013 |
| `gc_state_coverage_time.png` | file | 44173 |
| `gc_states.png` | file | 215129 |
| `gc_stdres_acf.png` | file | 13630 |
| `gc_stdres_hist.png` | file | 11278 |
| `gc_stdres_qq.png` | file | 25077 |
| `gc_stdres_time.png` | file | 81529 |
| `gc_y_fit.png` | file | 56049 |
| `metrics.csv` | file | 112 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 135 |
