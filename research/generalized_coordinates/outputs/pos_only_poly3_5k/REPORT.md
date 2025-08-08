# GC Scenario Report: pos_only_poly3_5k
Generated: 2025-08-07 20:30:47

## Scenario
- **n**: 5000
- **dt**: 0.05
- **σ_a**: 0.2
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `poly` with kwargs `Dict{Symbol, Any}(:degree => 3, :coeffs => [0.0, 0.5, -0.01, 0.0001])`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.065030 | 0.080395 | 0.141186 |
| coverage95 | 0.980 | 0.997 | 1.000 |

## Free energy over iterations
- iterations: 5000
- start: 10.452203, end: 254671020.017150, delta: 254671009.564947

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_poly3_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12339 |
| `gc_dashboard.png` | file | 136028 |
| `gc_derivative_consistency.png` | file | 43597 |
| `gc_errors.png` | file | 106829 |
| `gc_fe_cumsum.png` | file | 24011 |
| `gc_fe_iterations.png` | file | 27845 |
| `gc_free_energy_timeseries.csv` | file | 347388 |
| `gc_mse_time.png` | file | 144361 |
| `gc_residual_hist.png` | file | 12439 |
| `gc_residuals.png` | file | 54087 |
| `gc_rmse.png` | file | 13370 |
| `gc_scatter_true_vs_inferred.png` | file | 139378 |
| `gc_state_coverage_time.png` | file | 43910 |
| `gc_states.png` | file | 124893 |
| `gc_stdres_acf.png` | file | 14099 |
| `gc_stdres_hist.png` | file | 10455 |
| `gc_stdres_qq.png` | file | 26376 |
| `gc_stdres_time.png` | file | 79348 |
| `gc_y_fit.png` | file | 25748 |
| `metrics.csv` | file | 112 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 121 |
