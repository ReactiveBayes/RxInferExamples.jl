# GC Scenario Report: pos_only_poly5_5k
Generated: 2025-08-07 20:31:27

## Scenario
- **n**: 5000
- **dt**: 0.05
- **σ_a**: 0.25
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `poly` with kwargs `Dict{Symbol, Any}(:degree => 5, :coeffs => [0.0, 0.4, -0.01, 0.0001, -1.0e-6, 5.0e-9])`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.069395 | 0.087113 | 0.163568 |
| coverage95 | 0.966 | 0.990 | 1.000 |

## Free energy over iterations
- iterations: 5000
- start: 11.018457, end: 311226720.000540, delta: 311226708.982082

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_poly5_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12344 |
| `gc_dashboard.png` | file | 131651 |
| `gc_derivative_consistency.png` | file | 41800 |
| `gc_errors.png` | file | 81077 |
| `gc_fe_cumsum.png` | file | 23845 |
| `gc_fe_iterations.png` | file | 25322 |
| `gc_free_energy_timeseries.csv` | file | 347358 |
| `gc_mse_time.png` | file | 113655 |
| `gc_residual_hist.png` | file | 12417 |
| `gc_residuals.png` | file | 54737 |
| `gc_rmse.png` | file | 10768 |
| `gc_scatter_true_vs_inferred.png` | file | 102766 |
| `gc_state_coverage_time.png` | file | 46355 |
| `gc_states.png` | file | 118929 |
| `gc_stdres_acf.png` | file | 14386 |
| `gc_stdres_hist.png` | file | 10566 |
| `gc_stdres_qq.png` | file | 23979 |
| `gc_stdres_time.png` | file | 75723 |
| `gc_y_fit.png` | file | 24993 |
| `metrics.csv` | file | 111 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 122 |
