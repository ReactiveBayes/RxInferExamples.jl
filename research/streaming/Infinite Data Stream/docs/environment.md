### Environment and Observations

Defined in `environment.jl`:
- `Environment`: holds RNG, current state, observation precision, and buffers for `history` and `observations`.
- `getnext!`: increments time, computes hidden signal `10 * sin(0.1 * t)`, draws an observation from `NormalMeanPrecision(nextstate, observation_precision)`, and appends both to buffers.
- `gethistory` / `getobservations`: readers.

Parity with notebook:
- Same sinusoidal hidden state and NormalMeanPrecision observation model.
- `StableRNGs` used for reproducibility.

