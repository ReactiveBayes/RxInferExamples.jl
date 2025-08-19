include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using RxInfer
using .InfiniteDataStreamModel
using .InfiniteDataStreamEnv
using .InfiniteDataStreamUpdates
using .InfiniteDataStreamStreams

initial_state         = 0.0
observation_precision = 0.1
n = 300
interval_ms = 41

env = Environment(initial_state, observation_precision)

producer() = getnext!(env)
observations = timer_observations(interval_ms, n, producer)
datastream = observations |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d, ))

autoupdates = make_autoupdates()
init = make_initialization()

engine = infer(
    model          = kalman_filter(),
    constraints    = filter_constraints(),
    datastream     = datastream,
    autoupdates    = autoupdates,
    returnvars     = (:x_current, ),
    initialization = init,
    iterations     = 10,
    autostart      = false,
)

RxInfer.start(engine)

