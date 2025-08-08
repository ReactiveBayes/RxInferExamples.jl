# GC Scenario Report: pos_only_sin_f02_2000
Generated: 2025-08-08 10:24:02

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.1
- **σ_obs_pos**: 0.3
- **σ_obs_vel**: NaN
- **generator**: `sinusoid` with kwargs `Dict{Symbol, Any}(:freq => 0.2, :amp => 5.0)`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 |
|---|---:|---:|---:|---:|---:|
| rmse | 0.286725 | 0.386614 | 0.755746 | 0.151987 | 0.000464 |
| coverage95 | 0.174 | 0.179 | 0.348 | 0.160 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 16.973300, end: 111290531.280455, delta: 111290514.307155

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_5/pos_only_sin_f02_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 15706 |
| `gc_dashboard.png` | file | 253646 |
| `gc_derivative_consistency.png` | file | 37148 |
| `gc_errors.png` | file | 48320 |
| `gc_fe_cumsum.png` | file | 24325 |
| `gc_fe_iterations.png` | file | 30504 |
| `gc_free_energy_timeseries.csv` | file | 139078 |
| `gc_mse_time.png` | file | 140630 |
| `gc_residual_hist.png` | file | 12142 |
| `gc_residuals.png` | file | 65362 |
| `gc_rmse.png` | file | 13627 |
| `gc_scatter_true_vs_inferred.png` | file | 101377 |
| `gc_state_coverage_time.png` | file | 123807 |
| `gc_states.png` | file | 234029 |
| `gc_stdres_acf.png` | file | 12747 |
| `gc_stdres_hist.png` | file | 12939 |
| `gc_stdres_qq.png` | file | 27838 |
| `gc_stdres_time.png` | file | 81855 |
| `gc_y_fit.png` | file | 141864 |
| `metrics.csv` | file | 257 |
| `post_mean.csv` | file | 207104 |
| `post_var.csv` | file | 219997 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 139 |
| `x_true.csv` | file | 137630 |
| `y.csv` | file | 46793 |
