# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 10:23:18

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 |
|---|---:|---:|---:|---:|---:|
| rmse | 0.115430 | 0.140945 | 0.342200 | 0.105650 | 0.001942 |
| coverage95 | 0.961 | 0.971 | 0.959 | 1.000 | 0.875 |

## Free energy over iterations
- iterations: 2000
- start: 16.795773, end: 427431632.592041, delta: 427431615.796268

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_5/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 15894 |
| `gc_dashboard.png` | file | 142577 |
| `gc_derivative_consistency.png` | file | 36682 |
| `gc_errors.png` | file | 97039 |
| `gc_fe_cumsum.png` | file | 22496 |
| `gc_fe_iterations.png` | file | 27151 |
| `gc_free_energy_timeseries.csv` | file | 135800 |
| `gc_mse_time.png` | file | 174010 |
| `gc_residual_hist.png` | file | 9195 |
| `gc_residuals.png` | file | 69578 |
| `gc_rmse.png` | file | 13715 |
| `gc_scatter_true_vs_inferred.png` | file | 105293 |
| `gc_state_coverage_time.png` | file | 109020 |
| `gc_states.png` | file | 124046 |
| `gc_stdres_acf.png` | file | 15027 |
| `gc_stdres_hist.png` | file | 12096 |
| `gc_stdres_qq.png` | file | 26728 |
| `gc_stdres_time.png` | file | 93782 |
| `gc_y_fit.png` | file | 27021 |
| `metrics.csv` | file | 289 |
| `post_mean.csv` | file | 207413 |
| `post_var.csv` | file | 217147 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 146 |
| `x_true.csv` | file | 208132 |
| `y.csv` | file | 47602 |
