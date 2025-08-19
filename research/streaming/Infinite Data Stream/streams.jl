module InfiniteDataStreamStreams

using Rocket

export to_namedtuple_stream, timer_observations

to_namedtuple_stream(xs) = from(xs) |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d, ))

"""
timer_observations(interval_ms::Int, n::Int, producer::Function)

Create a Rocket timer that calls `producer()` every `interval_ms` milliseconds, taking `n` values.
The `producer` should return a Float64 observation.
"""
function timer_observations(interval_ms::Int, n::Int, producer::Function)
    return timer(interval_ms, interval_ms) |> map(Float64, _ -> producer()) |> take(n)
end

end # module

