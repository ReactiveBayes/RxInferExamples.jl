### Streams

Defined in `streams.jl` using `Rocket`:
- `to_namedtuple_stream(xs)`: wraps a vector of Float64 as a stream of named tuples `(y=...)` for RxInfer.
- `timer_observations(interval_ms, n, producer)`: creates a timer-based Observable that calls `producer()` every `interval_ms` ms and takes `n` samples.

Consistency:
- `run_static.jl` uses `to_namedtuple_stream(observations)` identical to the notebook static section.
- `run_realtime.jl` and `run.jl` use the timer pipeline analogous to the notebook’s realtime example.

