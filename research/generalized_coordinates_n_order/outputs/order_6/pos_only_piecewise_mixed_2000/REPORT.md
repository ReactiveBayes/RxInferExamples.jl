# GC Scenario Report: pos_only_piecewise_mixed_2000
Generated: 2025-08-08 10:26:53

## Scenario
- **n**: 2000
- **dt**: 0.05
- **σ_a**: 0.7
- **σ_obs_pos**: 0.4
- **σ_obs_vel**: NaN
- **generator**: `piecewise_mixed` with kwargs `Dict{Symbol, Any}(:segment_length => 500, :segments => [Dict(:freqs => [0.05, 0.12], :coeffs => [0.0, 0.15, 0.0, 0.0006, 0.0, -1.0e-6], :phases => [0.0, 1.0471975511965976], :amps => [6.0, 3.5]), Dict(:freqs => [0.4, 0.8, 1.2], :coeffs => [0.0, -0.1, 0.0, -0.0005, 0.0, 1.0e-6], :phases => [0.5235987755982988, 1.5707963267948966, 2.0943951023931953], :amps => [5.0, 3.0, 2.0]), Dict(:freqs => [0.2, 0.33, 0.5], :coeffs => [0.0, 0.05, 0.0, 0.0003, 0.0, -8.0e-7], :phases => [0.7853981633974483, 2.356194490192345, 0.39269908169872414], :amps => [4.0, 2.5, 1.5]), Dict(:freqs => [0.1, 0.25, 0.65], :coeffs => [0.0, 0.0, 0.0, -0.0002, 0.0, 5.0e-7], :phases => [1.5707963267948966, 0.6283185307179586, 2.748893571891069], :amps => [7.0, 3.0, 2.0])])`

## Metrics

| metric | dim_1 | dim_2 | dim_3 | dim_4 | dim_5 | dim_6 |
|---|---:|---:|---:|---:|---:|---:|
| rmse | 1.421718 | 7.713970 | 48.115131 | 0.604820 | 0.036100 | 0.001166 |
| coverage95 | 0.285 | 0.281 | 0.300 | 1.000 | 1.000 | 1.000 |

## Free energy over iterations
- iterations: 2000
- start: 19.623146, end: 1788232005.108654, delta: 1788231985.485508

## Outputs in `/Users/4d/Documents/GitHub/RxInferExamples.jl/research/generalized_coordinates_n_order/outputs/order_6/pos_only_piecewise_mixed_2000`

| name | type | bytes |
|---|---|---:|
| `gc_coverage.png` | file | 16539 |
| `gc_dashboard.png` | file | 242859 |
| `gc_derivative_consistency.png` | file | 41368 |
| `gc_errors.png` | file | 62711 |
| `gc_fe_cumsum.png` | file | 21255 |
| `gc_fe_iterations.png` | file | 31331 |
| `gc_free_energy_timeseries.csv` | file | 136416 |
| `gc_mse_time.png` | file | 163279 |
| `gc_residual_hist.png` | file | 10841 |
| `gc_residuals.png` | file | 56838 |
| `gc_rmse.png` | file | 14623 |
| `gc_scatter_true_vs_inferred.png` | file | 143680 |
| `gc_state_coverage_time.png` | file | 145704 |
| `gc_states.png` | file | 218962 |
| `gc_stdres_acf.png` | file | 12547 |
| `gc_stdres_hist.png` | file | 12993 |
| `gc_stdres_qq.png` | file | 25989 |
| `gc_stdres_time.png` | file | 63998 |
| `gc_y_fit.png` | file | 113381 |
| `metrics.csv` | file | 283 |
| `post_mean.csv` | file | 250716 |
| `post_var.csv` | file | 254771 |
| `rxinfer_free_energy.csv` | file | 2214 |
| `scenario_config.toml` | file | 154 |
| `x_true.csv` | file | 146974 |
| `y.csv` | file | 46628 |
