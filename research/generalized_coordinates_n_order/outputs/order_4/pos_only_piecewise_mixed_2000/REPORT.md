# GC Scenario Report: pos_only_piecewise_mixed_2000
Generated: 2025-08-08 09:34:16

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.7
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `piecewise_mixed` with kwargs `Dict{Symbol, Any}(:segment_length => 500, :segments => [Dict(:freqs => [0.05, 0.12], :coeffs => [0.0, 0.15, 0.0, 0.0006, 0.0, -1.0e-6], :phases => [0.0, 1.0471975511965976], :amps => [6.0, 3.5]), Dict(:freqs => [0.4, 0.8, 1.2], :coeffs => [0.0, -0.1, 0.0, -0.0005, 0.0, 1.0e-6], :phases => [0.5235987755982988, 1.5707963267948966, 2.0943951023931953], :amps => [5.0, 3.0, 2.0]), Dict(:freqs => [0.2, 0.33, 0.5], :coeffs => [0.0, 0.05, 0.0, 0.0003, 0.0, -8.0e-7], :phases => [0.7853981633974483, 2.356194490192345, 0.39269908169872414], :amps => [4.0, 2.5, 1.5]), Dict(:freqs => [0.1, 0.25, 0.65], :coeffs => [0.0, 0.0, 0.0, -0.0002, 0.0, 5.0e-7], :phases => [1.5707963267948966, 0.6283185307179586, 2.748893571891069], :amps => [7.0, 3.0, 2.0])])`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 |
|---|---:|---:|---:|---:|
| rmse | 1.423905 | 7.716362 | 48.118056 | 0.163626 |
| coverage95 | 0.302 | 0.286 | 0.298 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 13.368372, end: 902241864.961883, delta: 902241851.593510

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_4/pos_only_piecewise_mixed_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 15168 |
| `gc_dashboard.png` | file | 192136 |
| `gc_derivative_consistency.png` | file | 40017 |
| `gc_errors.png` | file | 62397 |
| `gc_fe_cumsum.png` | file | 21145 |
| `gc_fe_iterations.png` | file | 25044 |
| `gc_free_energy_timeseries.csv` | file | 137512 |
| `gc_mse_time.png` | file | 134004 |
| `gc_residual_hist.png` | file | 11021 |
| `gc_residuals.png` | file | 57745 |
| `gc_rmse.png` | file | 13319 |
| `gc_scatter_true_vs_inferred.png` | file | 111533 |
| `gc_state_coverage_time.png` | file | 129133 |
| `gc_states.png` | file | 177352 |
| `gc_stdres_acf.png` | file | 12674 |
| `gc_stdres_hist.png` | file | 12836 |
| `gc_stdres_qq.png` | file | 25900 |
| `gc_stdres_time.png` | file | 64656 |
| `gc_y_fit.png` | file | 114962 |
| `metrics.csv` | file | 147 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 154 |
