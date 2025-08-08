# GC Scenario Report: pos_only_constant_2000
Generated: 2025-08-08 09:33:47

## Scenario
- **n**: 2000
- **dt**: 0.1
- **σ_a**: 0.25
- **σ_obs_pos**: 0.5
- **σ_obs_vel**: NaN
- **generator**: `constant_accel` with kwargs `Dict{Symbol, Any}()`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 |
|---|---:|---:|---:|---:|
| rmse | 0.113832 | 0.149840 | 0.372056 | 0.097544 |
| coverage95 | 0.964 | 0.952 | 0.934 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 14.201451, end: 412623651.319942, delta: 412623637.118491

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_4/pos_only_constant_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 15251 |
| `gc_dashboard.png` | file | 110817 |
| `gc_derivative_consistency.png` | file | 41691 |
| `gc_errors.png` | file | 104333 |
| `gc_fe_cumsum.png` | file | 22521 |
| `gc_fe_iterations.png` | file | 25035 |
| `gc_free_energy_timeseries.csv` | file | 135917 |
| `gc_mse_time.png` | file | 177581 |
| `gc_residual_hist.png` | file | 9761 |
| `gc_residuals.png` | file | 62029 |
| `gc_rmse.png` | file | 13000 |
| `gc_scatter_true_vs_inferred.png` | file | 81189 |
| `gc_state_coverage_time.png` | file | 101065 |
| `gc_states.png` | file | 91490 |
| `gc_stdres_acf.png` | file | 14372 |
| `gc_stdres_hist.png` | file | 10753 |
| `gc_stdres_qq.png` | file | 25699 |
| `gc_stdres_time.png` | file | 82183 |
| `gc_y_fit.png` | file | 25938 |
| `metrics.csv` | file | 150 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 146 |
