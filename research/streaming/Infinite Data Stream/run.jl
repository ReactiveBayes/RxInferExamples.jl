using Dates
using Plots
using DelimitedFiles

# Ensure the example runs with its local project even if invoked without --project
try
    @eval begin
        import Pkg
        Pkg.activate(@__DIR__)
        Pkg.instantiate()
    end
catch
end

using Random, StableRNGs, RxInfer, Rocket

# Ensure headless GR to avoid blocking when encoding images/animations
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end

include("utils.jl")
using .InfiniteDataStreamUtils
InfiniteDataStreamUtils.load_modules!()
const IDS_CFG = InfiniteDataStreamUtils.load_config()

mkpath(get(IDS_CFG, "output_dir", "output"))
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath(get(IDS_CFG, "output_dir", "output"), ts)
static_dir = joinpath(outdir, "static")
realtime_dir = joinpath(outdir, "realtime")
cmp_dir = joinpath(outdir, "comparison")
mkpath(static_dir); mkpath(realtime_dir); mkpath(cmp_dir)

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
log("Writing artifacts under: $(outdir)")

# Helper: compute per-timestep FE by re-inferring on growing prefixes
function compute_per_timestep_fe(observations::Vector{Float64}; iterations::Int=10)
    fe_series = Float64[]
    total = length(observations)
    log_stride = try
        Int(get(IDS_CFG, "fe_log_stride", 10))
    catch
        10
    end
    start_t = time()
    log("[static] FE prefix re-inference: total=$(total), iters/step=$(iterations), log_stride=$(log_stride)")
    for t in 1:length(observations)
        ds_t = Main.InfiniteDataStreamStreams.to_namedtuple_stream(observations[1:t])
        au = Main.InfiniteDataStreamUpdates.make_autoupdates()
        init = Main.InfiniteDataStreamUpdates.make_initialization()
        eng_t = infer(
            model          = Main.InfiniteDataStreamModel.kalman_filter(),
            constraints    = Main.InfiniteDataStreamModel.filter_constraints(),
            datastream     = ds_t,
            autoupdates    = au,
            returnvars     = (:x_current,),
            initialization = init,
            iterations     = iterations,
            free_energy    = true,
            keephistory    = 1,
            autostart      = true,
        )
        push!(fe_series, eng_t.free_energy_history[end])
        if (t % log_stride == 0) || (t == total)
            elapsed = time() - start_t
            rate = elapsed / t
            eta_s = round(Int, max(0, rate * (total - t)))
            pct = round(100 * t / total; digits=1)
            log("[static] FE computed for t=$(t)/$(total) ($(pct)%); ETA ≈ $(eta_s)s")
        end
    end
    return fe_series
end

# Static case
begin
    using .InfiniteDataStreamModel
    using .InfiniteDataStreamEnv
    using .InfiniteDataStreamUpdates
    using .InfiniteDataStreamStreams
    using .InfiniteDataStreamViz

    initial_state         = Float64(get(IDS_CFG, "initial_state", 0.0))
    observation_precision = Float64(get(IDS_CFG, "observation_precision", 0.1))
    n = Int(get(IDS_CFG, "n", 300))

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

    # Optional animations controlled by config/env (default: enabled)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        log("[static] GIF generation enabled via IDS_MAKE_GIF=1")
        stride = Int(get(IDS_CFG, "gif_stride", 5))
        log("[static] rendering animation frames (stride=$(stride)) …")
        anim = @animate for i in 1:stride:n
            InfiniteDataStreamViz.plot_estimates(μ, σ2, history, observations; upto=i)
            if i % (50*stride) == 0 || i == n
                pct = round(100 * i / n; digits=1)
                @info "[static] rendered frame $i/$n ($(pct)%)"
            end
        end
        log("[static] saving GIF …")
        InfiniteDataStreamViz.save_gif(anim, joinpath(static_dir, "static_inference.gif"))
        log("[static] saved: $(joinpath(static_dir, "static_inference.gif"))")
        # Free-energy animation (per-timestep via prefix re-inference)
        fe_t = compute_per_timestep_fe(observations; iterations=10)
        anim_fe = InfiniteDataStreamViz.animate_free_energy(fe_t; stride=stride)
        static_fe_gif = joinpath(static_dir, "static_free_energy.gif")
        InfiniteDataStreamViz.save_gif(anim_fe, static_fe_gif)
        log("[static] saved: $(static_fe_gif)")
        # Composed animation (estimates + free energy)
        anim_comp = InfiniteDataStreamViz.animate_composed_estimates_fe(μ, σ2, history, observations, fe_t; stride=stride)
        static_comp_gif = joinpath(static_dir, "static_composed_estimates_fe.gif")
        InfiniteDataStreamViz.save_gif(anim_comp, static_comp_gif)
        log("[static] saved: $(static_comp_gif)")
        try
            cp(joinpath(static_dir, "static_free_energy.csv"), joinpath(cmp_dir, "static_free_energy.csv"), force=true)
            cp(static_fe_gif, joinpath(cmp_dir, "static_free_energy.gif"), force=true)
            cp(static_comp_gif, joinpath(cmp_dir, "static_composed_estimates_fe.gif"), force=true)
        catch
        end
    else
        log("[static] GIF generation disabled (set IDS_MAKE_GIF=1 to enable)")
    end

    log("[static] saving free-energy plot …")
    # Allow tests to skip the second FE prefix pass (already done above for GIF)
    if get(IDS_CFG, "skip_fe_replot", false)
        log("[static] skipping FE recomputation (IDS_SKIP_FE_REPLOT=1)")
        fe_t_plot = readdlm(joinpath(static_dir, "static_free_energy.csv"))[:]
    else
        fe_t_plot = compute_per_timestep_fe(observations; iterations=Int(get(IDS_CFG, "iterations", 10)))
        # Persist numeric outputs (CSV)
        writedlm(joinpath(static_dir, "static_free_energy.csv"), fe_t_plot)
    end
    pfe = plot(fe_t_plot; label="Bethe Free Energy (averaged)")
    png(pfe, joinpath(static_dir, "static_free_energy.png"))
    writedlm(joinpath(static_dir, "static_posterior_x_current.csv"), hcat(μ, σ2))
    writedlm(joinpath(static_dir, "static_truth_history.csv"), history)
    writedlm(joinpath(static_dir, "static_observations.csv"), observations)
    if tau_series !== nothing
        tau_shape = shape.(tau_series)
        tau_rate  = rate.(tau_series)
        tau_mean  = tau_shape ./ tau_rate
        writedlm(joinpath(static_dir, "static_posterior_tau_shape_rate.csv"), hcat(tau_shape, tau_rate))
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

    initial_state         = Float64(get(IDS_CFG, "initial_state", 0.0))
    observation_precision = Float64(get(IDS_CFG, "observation_precision", 0.1))
    n = Int(get(IDS_CFG, "n", 300))
    interval_ms = Int(get(IDS_CFG, "interval_ms", 41))

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
        free_energy    = true,
        autostart      = false,
    )

    # Subscribe to collect posterior snapshots for plotting later
    mu_rt = Float64[]
    var_rt = Float64[]
    fe_rt  = Float64[]
    tau_shape_rt = Float64[]
    tau_rate_rt  = Float64[]
    obs_count = Ref(0)
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
    # Count observations as they arrive to show progress
    _ = subscribe!(datastream, _ -> (obs_count[] += 1))
    # Streamed free-energy (if exposed by engine)
    try
        if hasproperty(engine, :free_energy) && engine.free_energy !== nothing
            _ = subscribe!(engine.free_energy, fe -> push!(fe_rt, Float64(fe)))
        end
    catch
    end

    log("[realtime] starting engine (interval=$(interval_ms)ms, n=$(n)) …")
    RxInfer.start(engine)
    # Simple progress while we wait for stream completion
    total = ceil(Int, n * interval_ms / 1000) + 1
    for s in 1:total
        sleep(1)
        if s % 2 == 0 || s == total
            pct = round(100 * min(obs_count[], n) / n; digits=1)
            log("[realtime] elapsed $(s)/$(total)s, observed=$(obs_count[]) / $(n) ($(pct)%)")
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
            stride_rt = Int(get(IDS_CFG, "rt_gif_stride", 5))
            anim_rt = @animate for i in 1:stride_rt:upto
                InfiniteDataStreamViz.plot_estimates(mu_rt[1:upto], var_rt[1:upto], hist, obs; upto=i)
            end
            InfiniteDataStreamViz.save_gif(anim_rt, joinpath(realtime_dir, "realtime_inference.gif"))
            log("[realtime] saved: $(joinpath(realtime_dir, "realtime_inference.gif"))")
            try
                cp(joinpath(static_dir, "static_free_energy.csv"), joinpath(realtime_dir, "realtime_free_energy.csv"), force=true)
                if isfile(joinpath(static_dir, "static_free_energy.gif"))
                    cp(joinpath(static_dir, "static_free_energy.gif"), joinpath(realtime_dir, "realtime_free_energy.gif"), force=true)
                    log("[realtime] mirrored FE GIF from static")
                end
            catch
            end
            # Persist realtime FE if captured
            if !isempty(fe_rt)
                writedlm(joinpath(realtime_dir, "realtime_free_energy.csv"), fe_rt)
                if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
                    anim_fe_rt = InfiniteDataStreamViz.animate_free_energy(fe_rt; stride=stride_rt)
                    InfiniteDataStreamViz.save_gif(anim_fe_rt, joinpath(realtime_dir, "realtime_free_energy.gif"))
                    log("[realtime] saved: $(joinpath(realtime_dir, "realtime_free_energy.gif"))")
                    # Two-panel composed realtime animation (estimates + FE)
                    anim_rt_comp = InfiniteDataStreamViz.animate_composed_estimates_fe(mu_rt[1:upto], var_rt[1:upto], hist, obs, fe_rt; stride=stride_rt)
                    InfiniteDataStreamViz.save_gif(anim_rt_comp, joinpath(realtime_dir, "realtime_composed_estimates_fe.gif"))
                    log("[realtime] saved: $(joinpath(realtime_dir, "realtime_composed_estimates_fe.gif"))")
                end
            elseif get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
                log("[realtime] FE stream not exposed; mirroring static FE artifacts")
            end
            # Realtime composed GIFs
            if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
                log("[realtime] GIF generation enabled via IDS_MAKE_GIF=1")
            else
                log("[realtime] GIF generation disabled (set IDS_MAKE_GIF=1 to enable)")
            end
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
        println(io, "free_energy_points_captured=$(length(fe_rt))")
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

    # Additional comparisons
    # 1) Scatter: static vs realtime means
    pscatter = scatter(μs[1:upto], μr[1:upto]; ms=3, alpha=0.7, label="points", xlabel="static μ", ylabel="realtime μ")
    plot!(pscatter, [minimum(μs[1:upto]); maximum(μs[1:upto])], [minimum(μs[1:upto]); maximum(μs[1:upto])]; label="y=x", color=:gray, lw=1.5)
    png(pscatter, joinpath(cmp_dir, "scatter_static_vs_realtime.png"))

    # 2) Residuals (truth - mean)
    r_static = static_truth[1:upto] .- μs[1:upto]
    r_rt     = rt_truth[1:upto] .- μr[1:upto]
    pr_s = plot(r_static; label="residual (static)", xlabel="t", ylabel="truth - mean", size=(1000,300))
    hline!(pr_s, [0.0]; color=:gray, lw=1, label="0")
    png(pr_s, joinpath(cmp_dir, "residuals_static.png"))
    pr_r = plot(r_rt; label="residual (realtime)", xlabel="t", ylabel="truth - mean", size=(1000,300))
    hline!(pr_r, [0.0]; color=:gray, lw=1, label="0")
    png(pr_r, joinpath(cmp_dir, "residuals_realtime.png"))

    # Optional comparison animation (overlay evolving)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        stride_cmp = parse(Int, get(ENV, "IDS_GIF_STRIDE", "5"))
        anim_overlay = InfiniteDataStreamViz.animate_overlay_means(static_truth, μs, μr; stride=stride_cmp)
        InfiniteDataStreamViz.save_gif(anim_overlay, joinpath(cmp_dir, "overlay_means.gif"))
    else
        @info "[comparison] overlay animation disabled (set IDS_MAKE_GIF=1 to enable)"
    end
catch e
    @warn "comparison report failed" error=e
end

