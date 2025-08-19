include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!(); const IDS_CFG = InfiniteDataStreamUtils.load_config()
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

initial_state         = Float64(get(IDS_CFG, "initial_state", 0.0))
observation_precision = Float64(get(IDS_CFG, "observation_precision", 0.1))
n = Int(get(IDS_CFG, "n", 300))

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
    keephistory    = Int(get(IDS_CFG, "keephistory", 10_000)),
    historyvars    = (x_current = KeepLast(), τ = KeepLast()),
    initialization = init,
    iterations     = Int(get(IDS_CFG, "iterations", 10)),
    free_energy    = true,
    autostart      = true,
)

estimated = engine.history[:x_current]

# Default: save a single PNG snapshot (fast, non-blocking)
μ = mean.(estimated)
σ2 = var.(estimated)
p = InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=n)
png(p, "static_inference.png")

if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", false) ? "1" : "0") == "1"
    stride = Int(get(IDS_CFG, "gif_stride", 5))
    anim = @animate for i in 1:stride:n
        InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=i)
    end
    InfiniteDataStreamViz.save_gif(anim, "infinite-data-stream-inference.gif")
    # FE per-timestep (prefix re-inference)
    fe_series = let obs = observations, iters = Int(get(IDS_CFG, "iterations", 10))
        fe = Float64[]
        for t in 1:length(obs)
            ds_t = to_namedtuple_stream(obs[1:t])
            eng_t = infer(
                model          = kalman_filter(),
                constraints    = filter_constraints(),
                datastream     = ds_t,
                autoupdates    = autoupdates,
                returnvars     = (:x_current,),
                initialization = init,
                iterations     = iters,
                free_energy    = true,
                autostart      = true,
            )
            push!(fe, eng_t.free_energy_history[end])
        end
        fe
    end
    anim_fe = @animate for i in 1:stride:length(fe_series)
        plot(fe_series[1:i]; label="Bethe Free Energy (avg)", xlabel="t")
    end
    InfiniteDataStreamViz.save_gif(anim_fe, "static_free_energy.gif")
end

plot(engine.free_energy_history; label="Bethe Free Energy (averaged)")

