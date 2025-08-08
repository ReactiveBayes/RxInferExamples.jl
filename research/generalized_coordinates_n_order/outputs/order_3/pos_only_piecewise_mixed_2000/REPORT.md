# GC Scenario Report: pos_only_piecewise_mixed_2000
Generated: 2025-08-08 10:22:03

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.7
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `piecewise_mixed` with kwargs `Dict{Symbol, Any}(:segment_length => 500, :segments => [Dict(:freqs => [0.05, 0.12], :coeffs => [0.0, 0.15, 0.0, 0.0006, 0.0, -1.0e-6], :phases => [0.0, 1.0471975511965976], :amps => [6.0, 3.5]), Dict(:freqs => [0.4, 0.8, 1.2], :coeffs => [0.0, -0.1, 0.0, -0.0005, 0.0, 1.0e-6], :phases => [0.5235987755982988, 1.5707963267948966, 2.0943951023931953], :amps => [5.0, 3.0, 2.0]), Dict(:freqs => [0.2, 0.33, 0.5], :coeffs => [0.0, 0.05, 0.0, 0.0003, 0.0, -8.0e-7], :phases => [0.7853981633974483, 2.356194490192345, 0.39269908169872414], :amps => [4.0, 2.5, 1.5]), Dict(:freqs => [0.1, 0.25, 0.65], :coeffs => [0.0, 0.0, 0.0, -0.0002, 0.0, 5.0e-7], :phases => [1.5707963267948966, 0.6283185307179586, 2.748893571891069], :amps => [7.0, 3.0, 2.0])])`

## Metrics

| metric | dim_1 | dim_2 | dim_3 |
|---|---:|---:|---:|
| rmse | 1.418153 | 7.711925 | 48.114650 |
| coverage95 | 0.304 | 0.282 | 0.300 |

## Free energy over iterations
- iterations: 2000
- start: 10.492725, end: 800146893.515096, delta: 800146883.022371

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_3/pos_only_piecewise_mixed_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 14295 |
| `gc_dashboard.png` | file | 179928 |
| `gc_derivative_consistency.png` | file | 40968 |
| `gc_errors.png` | file | 61825 |
| `gc_fe_cumsum.png` | file | 22838 |
| `gc_fe_iterations.png` | file | 25015 |
| `gc_free_energy_timeseries.csv` | file | 135548 |
| `gc_mse_time.png` | file | 112167 |
| `gc_residual_hist.png` | file | 10834 |
| `gc_residuals.png` | file | 58256 |
| `gc_rmse.png` | file | 12435 |
| `gc_scatter_true_vs_inferred.png` | file | 101398 |
| `gc_state_coverage_time.png` | file | 118754 |
| `gc_states.png` | file | 165634 |
| `gc_stdres_acf.png` | file | 12843 |
| `gc_stdres_hist.png` | file | 12867 |
| `gc_stdres_qq.png` | file | 25642 |
| `gc_stdres_time.png` | file | 65253 |
| `gc_y_fit.png` | file | 115985 |
| `metrics.csv` | file | 181 |
| `post_mean.csv` | file | 122908 |
| `post_var.csv` | file | 128854 |
| `rxinfer_free_energy.csv` | file | 2114 |
| `scenario_config.toml` | file | 154 |
| `x_true.csv` | file | 122956 |
| `y.csv` | file | 46666 |
