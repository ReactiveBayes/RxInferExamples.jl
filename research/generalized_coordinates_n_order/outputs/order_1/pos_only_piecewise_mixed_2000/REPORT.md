# GC Scenario Report: pos_only_piecewise_mixed_2000
Generated: 2025-08-08 10:21:07

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.7
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `piecewise_mixed` with kwargs `Dict{Symbol, Any}(:segment_length => 500, :segments => [Dict(:freqs => [0.05, 0.12], :coeffs => [0.0, 0.15, 0.0, 0.0006, 0.0, -1.0e-6], :phases => [0.0, 1.0471975511965976], :amps => [6.0, 3.5]), Dict(:freqs => [0.4, 0.8, 1.2], :coeffs => [0.0, -0.1, 0.0, -0.0005, 0.0, 1.0e-6], :phases => [0.5235987755982988, 1.5707963267948966, 2.0943951023931953], :amps => [5.0, 3.0, 2.0]), Dict(:freqs => [0.2, 0.33, 0.5], :coeffs => [0.0, 0.05, 0.0, 0.0003, 0.0, -8.0e-7], :phases => [0.7853981633974483, 2.356194490192345, 0.39269908169872414], :amps => [4.0, 2.5, 1.5]), Dict(:freqs => [0.1, 0.25, 0.65], :coeffs => [0.0, 0.0, 0.0, -0.0002, 0.0, 5.0e-7], :phases => [1.5707963267948966, 0.6283185307179586, 2.748893571891069], :amps => [7.0, 3.0, 2.0])])`

## Metrics

| metric | dim_1 |
|---|---:|
| rmse | 4.808360 |
| coverage95 | 0.002 |

## Free energy over iterations
- iterations: 2000
- start: 14.479560, end: 80705.174307, delta: 80690.694748

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_1/pos_only_piecewise_mixed_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 11467 |
| `gc_dashboard.png` | file | 69536 |
| `gc_derivative_consistency.png` | file | 15338 |
| `gc_errors.png` | file | 57946 |
| `gc_fe_cumsum.png` | file | 21518 |
| `gc_fe_iterations.png` | file | 24348 |
| `gc_free_energy_timeseries.csv` | file | 125313 |
| `gc_mse_time.png` | file | 44633 |
| `gc_residual_hist.png` | file | 11668 |
| `gc_residuals.png` | file | 51823 |
| `gc_rmse.png` | file | 8837 |
| `gc_scatter_true_vs_inferred.png` | file | 18195 |
| `gc_state_coverage_time.png` | file | 15425 |
| `gc_states.png` | file | 45093 |
| `gc_stdres_acf.png` | file | 13316 |
| `gc_stdres_hist.png` | file | 13994 |
| `gc_stdres_qq.png` | file | 26866 |
| `gc_stdres_time.png` | file | 56243 |
| `gc_y_fit.png` | file | 105897 |
| `metrics.csv` | file | 83 |
| `post_mean.csv` | file | 46671 |
| `post_var.csv` | file | 50600 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 154 |
| `x_true.csv` | file | 46683 |
| `y.csv` | file | 46635 |
