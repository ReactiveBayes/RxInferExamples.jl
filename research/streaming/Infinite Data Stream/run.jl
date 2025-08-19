using Dates
using Plots
using DelimitedFiles

# Ensure headless GR to avoid blocking when encoding images/animations
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end

include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()

mkpath("output")
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath("output", ts)
mkpath(outdir)

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
log("Writing artifacts under: $(outdir)")

# Static case
begin
    using RxInfer
    using .InfiniteDataStreamModel
    using .InfiniteDataStreamEnv
    using .InfiniteDataStreamUpdates
    using .InfiniteDataStreamStreams
    using .InfiniteDataStreamViz

    initial_state         = 0.0
    observation_precision = 0.1
    n = 300

    log("[static] generating $(n) observations …")
    env = Environment(initial_state, observation_precision)
    for i in 1:n
        getnext!(env)
        if i % 50 == 0
            log("[static] generated $(i)/$(n)")
        end
    end

    history = gethistory(env)
    observations = getobservations(env)
    datastream = to_namedtuple_stream(observations)

    autoupdates = make_autoupdates()
    init = make_initialization()

    log("[static] starting inference …")
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
    # Fast snapshot (non-animated) by default
    μ = mean.(estimated)
    σ2 = var.(estimated)
    p = InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=n)
    png(p, joinpath(outdir, "static_inference.png"))

    # Optional animation controlled by ENV flags
    if get(ENV, "IDS_MAKE_GIF", "0") == "1"
        stride = parse(Int, get(ENV, "IDS_GIF_STRIDE", "5"))
        log("[static] rendering animation frames (stride=$(stride)) …")
        anim = @animate for i in 1:stride:n
            InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=i)
            if i % (50*stride) == 0
                @info "[static] rendered frame $i/$n"
            end
        end
        log("[static] saving GIF …")
        InfiniteDataStreamViz.save_gif(anim, joinpath(outdir, "static_inference.gif"))
    end

    log("[static] saving free-energy plot …")
    pfe = plot(engine.free_energy_history; label="Bethe Free Energy (averaged)")
    png(pfe, joinpath(outdir, "static_free_energy.png"))
    # Persist numeric outputs (CSV)
    writedlm(joinpath(outdir, "static_free_energy.csv"), engine.free_energy_history)
    writedlm(joinpath(outdir, "static_posterior_x_current.csv"), hcat(μ, σ2))
    writedlm(joinpath(outdir, "static_truth_history.csv"), history)
    writedlm(joinpath(outdir, "static_observations.csv"), observations)
    log("[static] done")
end

# Realtime case
begin
    using RxInfer, Rocket
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

    log("[realtime] building engine …")
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

    # Subscribe to collect posterior snapshots for plotting later
    mu_rt = Float64[]
    var_rt = Float64[]
    _ = subscribe!(engine.posteriors[:x_current], q_current -> begin
        push!(mu_rt, mean(q_current))
        push!(var_rt, var(q_current))
    end)

    log("[realtime] starting engine (interval=$(interval_ms)ms, n=$(n)) …")
    RxInfer.start(engine)
    # Simple progress while we wait for stream completion
    total = ceil(Int, n * interval_ms / 1000) + 1
    for s in 1:total
        sleep(1)
        if s % 5 == 0 || s == total
            log("[realtime] elapsed $(s)/$(total) seconds …")
        end
    end
    # Stop engine explicitly (defensive) and persist artifacts
    try
        RxInfer.stop(engine)
    catch
    end

    # Save realtime posterior snapshot plot and CSVs
    let m = length(mu_rt)
        if m > 0
            # Align lengths in case timer delivered slightly fewer points
            hist = gethistory(env)
            obs  = getobservations(env)
            upto = min(m, length(hist), length(obs))
            p_rt = InfiniteDataStreamViz.plot_estimates(mu_rt[1:upto], var_rt[1:upto], hist, obs; upto=upto)
            png(p_rt, joinpath(outdir, "realtime_inference.png"))
            writedlm(joinpath(outdir, "realtime_posterior_x_current.csv"), hcat(mu_rt[1:upto], var_rt[1:upto]))
        end
    end
    writedlm(joinpath(outdir, "realtime_truth_history.csv"), gethistory(env))
    writedlm(joinpath(outdir, "realtime_observations.csv"), getobservations(env))

    open(joinpath(outdir, "realtime_summary.txt"), "w") do io
        println(io, "realtime_run_completed=true at ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "n=$(n), interval_ms=$(interval_ms)")
        println(io, "posterior_samples_captured=$(length(mu_rt))")
    end
    log("[realtime] done")
end

log("All outputs saved to $(outdir)")

