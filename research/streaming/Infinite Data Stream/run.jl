using Dates
using Plots
using DelimitedFiles
using Random, StableRNGs, RxInfer, Rocket

# Ensure headless GR to avoid blocking when encoding images/animations
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end

include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()

mkpath("output")
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath("output", ts)
static_dir = joinpath(outdir, "static")
realtime_dir = joinpath(outdir, "realtime")
cmp_dir = joinpath(outdir, "comparison")
mkpath(static_dir); mkpath(realtime_dir); mkpath(cmp_dir)

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
log("Writing artifacts under: $(outdir)")

# Static case
begin
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
    tau_series = get(engine.history, :τ, nothing)
    # Fast snapshot (non-animated) by default
    μ = mean.(estimated)
    σ2 = var.(estimated)
    p = InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=n)
    png(p, joinpath(static_dir, "static_inference.png"))

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
        InfiniteDataStreamViz.save_gif(anim, joinpath(static_dir, "static_inference.gif"))
    end

    log("[static] saving free-energy plot …")
    pfe = plot(engine.free_energy_history; label="Bethe Free Energy (averaged)")
    png(pfe, joinpath(static_dir, "static_free_energy.png"))
    # Persist numeric outputs (CSV)
    writedlm(joinpath(static_dir, "static_free_energy.csv"), engine.free_energy_history)
    writedlm(joinpath(static_dir, "static_posterior_x_current.csv"), hcat(μ, σ2))
    writedlm(joinpath(static_dir, "static_truth_history.csv"), history)
    writedlm(joinpath(static_dir, "static_observations.csv"), observations)
    if tau_series !== nothing
        tau_shape = shape.(tau_series)
        tau_rate  = rate.(tau_series)
        tau_mean  = tau_shape ./ tau_rate
        writedlm(joinpath(outdir, "static_posterior_tau_shape_rate.csv"), hcat(tau_shape, tau_rate))
        pτ = plot(tau_mean; label="E[τ]", xlabel="t", ylabel="precision")
        png(pτ, joinpath(static_dir, "static_tau_mean.png"))
    end
    log("[static] done")
end

# Realtime case
begin
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
    tau_shape_rt = Float64[]
    tau_rate_rt  = Float64[]
    _ = subscribe!(engine.posteriors[:x_current], q_current -> begin
        push!(mu_rt, mean(q_current))
        push!(var_rt, var(q_current))
    end)
    # τ (observation precision) time series
    if haskey(engine.posteriors, :τ)
        _ = subscribe!(engine.posteriors[:τ], qτ -> begin
            push!(tau_shape_rt, shape(qτ))
            push!(tau_rate_rt, rate(qτ))
        end)
    end

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
            png(p_rt, joinpath(realtime_dir, "realtime_inference.png"))
            # Realtime GIF (ensure present by default, stride to limit cost)
            stride_rt = parse(Int, get(ENV, "IDS_RT_GIF_STRIDE", "5"))
            anim_rt = @animate for i in 1:stride_rt:upto
                InfiniteDataStreamViz.plot_estimates(mu_rt[1:upto], var_rt[1:upto], hist, obs; upto=i)
            end
            InfiniteDataStreamViz.save_gif(anim_rt, joinpath(realtime_dir, "realtime_inference.gif"))
            writedlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"), hcat(mu_rt[1:upto], var_rt[1:upto]))
            if !isempty(tau_shape_rt)
                writedlm(joinpath(realtime_dir, "realtime_posterior_tau_shape_rate.csv"), hcat(tau_shape_rt[1:upto], tau_rate_rt[1:upto]))
                tau_mean_rt = tau_shape_rt[1:upto] ./ tau_rate_rt[1:upto]
                pτrt = plot(tau_mean_rt; label="E[τ] realtime", xlabel="t", ylabel="precision")
                png(pτrt, joinpath(realtime_dir, "realtime_tau_mean.png"))
            end
        end
    end
    writedlm(joinpath(realtime_dir, "realtime_truth_history.csv"), gethistory(env))
    writedlm(joinpath(realtime_dir, "realtime_observations.csv"), getobservations(env))

    open(joinpath(realtime_dir, "realtime_summary.txt"), "w") do io
        println(io, "realtime_run_completed=true at ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "n=$(n), interval_ms=$(interval_ms)")
        println(io, "posterior_samples_captured=$(length(mu_rt))")
    end
    log("[realtime] done")
end

log("All outputs saved to $(outdir)")

# Comparison report (errors and overlay plots)
try
    # Load static
    static_truth = readdlm(joinpath(static_dir, "static_truth_history.csv"))[:]
    static_est   = readdlm(joinpath(static_dir, "static_posterior_x_current.csv"))
    μs = static_est[:,1]

    # Load realtime
    rt_truth = readdlm(joinpath(realtime_dir, "realtime_truth_history.csv"))[:]
    rt_est   = readdlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"))
    μr = rt_est[:,1]

    upto = min(length(static_truth), length(μs), length(rt_truth), length(μr))
    mse_static = mean((μs[1:upto] .- static_truth[1:upto]).^2)
    mse_rt     = mean((μr[1:upto] .- rt_truth[1:upto]).^2)
    mae_static = mean(abs.(μs[1:upto] .- static_truth[1:upto]))
    mae_rt     = mean(abs.(μr[1:upto] .- rt_truth[1:upto]))

    # Save metrics
    open(joinpath(cmp_dir, "metrics.txt"), "w") do io
        println(io, "mse_static=", mse_static)
        println(io, "mse_realtime=", mse_rt)
        println(io, "mae_static=", mae_static)
        println(io, "mae_realtime=", mae_rt)
        println(io, "n_compare=", upto)
    end

    # Overlay plot
    pcmp = plot(static_truth[1:upto]; label="truth", color=:black)
    plot!(pcmp, μs[1:upto]; label="static μ", color=:blue)
    plot!(pcmp, μr[1:upto]; label="realtime μ", color=:orange)
    png(pcmp, joinpath(cmp_dir, "means_compare.png"))
catch e
    @warn "comparison report failed" error=e
end

