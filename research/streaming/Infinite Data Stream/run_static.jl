include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using RxInfer, Plots

# Ensure headless, non-interactive GR to avoid blocking in servers/CI
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end
using .InfiniteDataStreamModel
using .InfiniteDataStreamEnv
using .InfiniteDataStreamUpdates
using .InfiniteDataStreamStreams
using .InfiniteDataStreamViz

initial_state         = 0.0
observation_precision = 0.1
n = 300

env = Environment(initial_state, observation_precision)
for _ in 1:n
    getnext!(env)
end

history = gethistory(env)
observations = getobservations(env)
datastream = to_namedtuple_stream(observations)

autoupdates = make_autoupdates()
init = make_initialization()

engine = infer(
    model          = kalman_filter(),
    constraints    = filter_constraints(),
    datastream     = datastream,
    autoupdates    = autoupdates,
    returnvars     = (:x_current, ),
    keephistory    = 10_000,
    historyvars    = (x_current = KeepLast(), τ = KeepLast()),
    initialization = init,
    iterations     = 10,
    free_energy    = true,
    autostart      = true,
)

estimated = engine.history[:x_current]

# Default: save a single PNG snapshot (fast, non-blocking)
μ = mean.(estimated)
σ2 = var.(estimated)
p = InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=n)
png(p, "static_inference.png")

# Optional: create an animated GIF when explicitly requested
if get(ENV, "IDS_MAKE_GIF", "0") == "1"
    stride = parse(Int, get(ENV, "IDS_GIF_STRIDE", "5"))
    anim = @animate for i in 1:stride:n
        InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=i)
    end
    InfiniteDataStreamViz.save_gif(anim, "infinite-data-stream-inference.gif")
end

plot(engine.free_energy_history; label="Bethe Free Energy (averaged)")

