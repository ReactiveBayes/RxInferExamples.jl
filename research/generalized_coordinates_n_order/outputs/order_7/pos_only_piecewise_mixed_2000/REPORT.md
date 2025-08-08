# GC Scenario Report: pos_only_piecewise_mixed_2000
Generated: 2025-08-08 10:29:06

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.7
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `piecewise_mixed` with kwargs `Dict{Symbol, Any}(:segment_length => 500, :segments => [Dict(:freqs => [0.05, 0.12], :coeffs => [0.0, 0.15, 0.0, 0.0006, 0.0, -1.0e-6], :phases => [0.0, 1.0471975511965976], :amps => [6.0, 3.5]), Dict(:freqs => [0.4, 0.8, 1.2], :coeffs => [0.0, -0.1, 0.0, -0.0005, 0.0, 1.0e-6], :phases => [0.5235987755982988, 1.5707963267948966, 2.0943951023931953], :amps => [5.0, 3.0, 2.0]), Dict(:freqs => [0.2, 0.33, 0.5], :coeffs => [0.0, 0.05, 0.0, 0.0003, 0.0, -8.0e-7], :phases => [0.7853981633974483, 2.356194490192345, 0.39269908169872414], :amps => [4.0, 2.5, 1.5]), Dict(:freqs => [0.1, 0.25, 0.65], :coeffs => [0.0, 0.0, 0.0, -0.0002, 0.0, 5.0e-7], :phases => [1.5707963267948966, 0.6283185307179586, 2.748893571891069], :amps => [7.0, 3.0, 2.0])])`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 | dim_7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rmse | 1.419916 | 7.715200 | 48.123444 | 0.865150 | 0.114661 | 0.009295 | 0.000458 |
| coverage95 | 0.303 | 0.281 | 0.300 | 0.917 | 0.818 | 0.825 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 23.043812, end: 3305800427.954789, delta: 3305800404.910977

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_7/pos_only_piecewise_mixed_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 17280 |
| `gc_dashboard.png` | file | 262400 |
| `gc_derivative_consistency.png` | file | 40371 |
| `gc_errors.png` | file | 63211 |
| `gc_fe_cumsum.png` | file | 22932 |
| `gc_fe_iterations.png` | file | 32447 |
| `gc_free_energy_timeseries.csv` | file | 136056 |
| `gc_mse_time.png` | file | 171375 |
| `gc_residual_hist.png` | file | 10734 |
| `gc_residuals.png` | file | 58182 |
| `gc_rmse.png` | file | 14998 |
| `gc_scatter_true_vs_inferred.png` | file | 152415 |
| `gc_state_coverage_time.png` | file | 160968 |
| `gc_states.png` | file | 236372 |
| `gc_stdres_acf.png` | file | 12722 |
| `gc_stdres_hist.png` | file | 12711 |
| `gc_stdres_qq.png` | file | 25750 |
| `gc_stdres_time.png` | file | 63116 |
| `gc_y_fit.png` | file | 114580 |
| `metrics.csv` | file | 327 |
| `post_mean.csv` | file | 294857 |
| `post_var.csv` | file | 295750 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 154 |
| `x_true.csv` | file | 154980 |
| `y.csv` | file | 46663 |
