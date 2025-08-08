# GC Scenario Report: pos_only_sin_mixed_5k
Generated: 2025-08-07 20:30:04

## Scenario
- **n**: 5000
- **dt**: 0.04
- **σ_a**: 0.15
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid_mixed` with kwargs `Dict{Symbol, Any}(:freqs => [0.1, 0.35, 0.6], :amps => [2.0, 1.0, 0.8])`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.611769 | 2.128475 | 7.756613 |
| coverage95 | 0.098 | 0.039 | 0.029 |

## Free energy over iterations
- iterations: 5000
- start: 17.880446, end: 142271518.812824, delta: 142271500.932378

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_sin_mixed_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12284 |
| `gc_dashboard.png` | file | 255932 |
| `gc_derivative_consistency.png` | file | 41485 |
| `gc_errors.png` | file | 115583 |
| `gc_fe_cumsum.png` | file | 24202 |
| `gc_fe_iterations.png` | file | 27175 |
| `gc_free_energy_timeseries.csv` | file | 347727 |
| `gc_mse_time.png` | file | 221340 |
| `gc_residual_hist.png` | file | 10511 |
| `gc_residuals.png` | file | 79314 |
| `gc_rmse.png` | file | 9410 |
| `gc_scatter_true_vs_inferred.png` | file | 117504 |
| `gc_state_coverage_time.png` | file | 127796 |
| `gc_states.png` | file | 271549 |
| `gc_stdres_acf.png` | file | 14604 |
| `gc_stdres_hist.png` | file | 12629 |
| `gc_stdres_qq.png` | file | 27434 |
| `gc_stdres_time.png` | file | 97134 |
| `gc_y_fit.png` | file | 204174 |
| `metrics.csv` | file | 109 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 136 |
