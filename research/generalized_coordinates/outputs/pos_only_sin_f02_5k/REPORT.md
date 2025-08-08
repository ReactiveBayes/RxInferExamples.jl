# GC Scenario Report: pos_only_sin_f02_5k
Generated: 2025-08-07 20:29:17

## Scenario
- **n**: 5000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | pos | vel | acc |
|---|---:|---:|---:|
| rmse | 0.288649 | 0.371492 | 0.602840 |
| coverage95 | 0.169 | 0.177 | 0.363 |

## Free energy over iterations
- iterations: 5000
- start: 10.250841, end: 100051484.950670, delta: 100051474.699829

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates/outputs/pos_only_sin_f02_5k`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 12303 |
| `gc_dashboard.png` | file | 333884 |
| `gc_derivative_consistency.png` | file | 36806 |
| `gc_errors.png` | file | 48462 |
| `gc_fe_cumsum.png` | file | 22927 |
| `gc_fe_iterations.png` | file | 26874 |
| `gc_free_energy_timeseries.csv` | file | 342273 |
| `gc_mse_time.png` | file | 103402 |
| `gc_residual_hist.png` | file | 12125 |
| `gc_residuals.png` | file | 58119 |
| `gc_rmse.png` | file | 11804 |
| `gc_scatter_true_vs_inferred.png` | file | 72157 |
| `gc_state_coverage_time.png` | file | 116442 |
| `gc_states.png` | file | 276207 |
| `gc_stdres_acf.png` | file | 12370 |
| `gc_stdres_hist.png` | file | 12974 |
| `gc_stdres_qq.png` | file | 27942 |
| `gc_stdres_time.png` | file | 79253 |
| `gc_y_fit.png` | file | 277111 |
| `metrics.csv` | file | 111 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 127 |
