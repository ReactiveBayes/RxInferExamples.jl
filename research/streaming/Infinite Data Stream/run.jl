using Dates
using Plots
using DelimitedFiles

#
# Infinite Data Stream: static vs realtime inference and Free Energy (FE)
#
# Static mode
# - Computes estimates for the full dataset and then records FE via a "prefix re-inference" method
#   (compute_per_timestep_fe): re-runs short inference on prefixes 1..T to obtain a per-step FE series.
# - This is intentional for batch analysis and does not use a live FE stream.
# - Artifacts: static_inference.png, static_inference.gif (optional), static_free_energy.csv/png/gif,
#   static_composed_estimates_fe.gif, and diagnostics CSVs.
#
# Realtime mode
# - Builds an engine with free_energy=true, keephistory>0; subscribes to a live FE stream if the engine
#   exposes an observable at `engine.free_energy` (or `engine.free_energy_stream`).
# - If a live FE stream is present, we persist exactly that stream as realtime_free_energy.csv and related
#   visuals. If not, we fallback to an online strict FE approximation computed on-the-fly from the observed
#   prefix, using the realtime iteration budget.
# - Artifacts: realtime_inference.png/gif, realtime_free_energy.csv/png/gif (from the live stream when
#   available, otherwise fallback series), realtime_composed_estimates_fe.gif, and diagnostics CSVs.
# - Logging explicitly reports whether the live FE stream subscription is active or if fallback is used.
#
# For deeper details and diagrams, see REALTIME_IN_RXINFER.md in this directory.
#
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

include("engine_wrapper.jl")
using .InfiniteDataStreamEngineWrapper

# Ensure headless GR to avoid blocking when encoding images/animations
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end

include("utils.jl")
using .InfiniteDataStreamUtils
InfiniteDataStreamUtils.load_modules!()
const IDS_CFG = InfiniteDataStreamUtils.load_config()

# Simple CLI overrides: accept pairs like --n 2000 --interval_ms 5 --rt_iterations 16 --rt_fe_every 1 --make_gif false
let i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if startswith(arg, "--") && i < length(ARGS)
            key = String(Symbol(replace(arg[3:end], '-' => '_')))
            val = ARGS[i+1]
            # Normalize and parse common keys
            if key in ("n","interval_ms","iterations","rt_iterations","keephistory","gif_stride","rt_gif_stride","seed","rt_fe_every")
                try IDS_CFG[key] = parse(Int, val) catch; end
            elseif key in ("make_gif",)
                IDS_CFG["make_gif"] = lowercase(val) in ("1","true","t","yes","y")
            elseif key == "output_dir"
                IDS_CFG["output_dir"] = val
            end
            i += 2
            continue
        end
        i += 1
    end
end

mkpath(get(IDS_CFG, "output_dir", "output"))
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath(get(IDS_CFG, "output_dir", "output"), ts)
static_dir = joinpath(outdir, "static")
realtime_dir = joinpath(outdir, "realtime")
cmp_dir = joinpath(outdir, "comparison")
mkpath(static_dir); mkpath(realtime_dir); mkpath(cmp_dir)

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
log("Writing artifacts under: $(outdir)")
log("Config: n=$(Int(get(IDS_CFG, "n", 0))), interval_ms=$(Int(get(IDS_CFG, "interval_ms", 0))), iterations=$(Int(get(IDS_CFG, "iterations", 0))), seed=$(Int(get(IDS_CFG, "seed", 0)))")

# Plot safety helper to avoid GR/GKS "SET_WINDOW" precision warnings
# Ensures finite limits and adds minimal padding when series is flat
function _apply_safe_limits!(plt, ys::AbstractVector{<:Real})
    finite_vals = filter(isfinite, ys)
    if isempty(finite_vals)
        try
            xlims!(plt, (0.0, 1.0)); ylims!(plt, (0.0, 1.0))
        catch; end
        return plt
    end
    ymin = minimum(finite_vals)
    ymax = maximum(finite_vals)
    # Ensure non-zero height window
    pad = max(1e-9, (ymax - ymin) * 0.05)
    if !isfinite(pad) || pad <= 0
        pad = 1e-6
    end
    yl = (ymin - pad, ymax + pad)
    try
        ylims!(plt, yl)
    catch; end
    # Set x-limits based on length if not already set explicitly
    try
        xl = (1.0, max(1.0, float(length(ys))))
        xlims!(plt, xl)
    catch; end
    return plt
end

# Phase timings for provenance
const _timings = Dict{String,Float64}()
function _tstart(name::String)
    _timings[name] = time()
end
function _tstop(name::String)
    if haskey(_timings, name)
        _timings[name] = time() - _timings[name]
    end
end

# Write a minimal provenance file (JSON-like) for reproducibility
open(joinpath(outdir, "provenance.json"), "w") do io
    println(io, "{")
    println(io, "  \"timestamp\": \"$(ts)\",")
    println(io, "  \"n\": $(Int(get(IDS_CFG, "n", 0))),")
    println(io, "  \"interval_ms\": $(Int(get(IDS_CFG, "interval_ms", 0))),")
    println(io, "  \"iterations\": $(Int(get(IDS_CFG, "iterations", 0))),")
    println(io, "  \"rt_iterations\": $(Int(get(IDS_CFG, "rt_iterations", get(IDS_CFG, "iterations", 0)))),")
    println(io, "  \"keephistory\": $(Int(get(IDS_CFG, "keephistory", 0))),")
    println(io, "  \"rt_fe_every\": $(Int(get(IDS_CFG, "rt_fe_every", 1))),")
    # Try to capture git metadata if available
    try
        git_commit = try readchomp(`git rev-parse --short HEAD`) catch; "" end
        git_branch = try readchomp(`git rev-parse --abbrev-ref HEAD`) catch; "" end
        git_status = try read(`git status --porcelain`, String) catch; "" end
        git_dirty = git_status != ""
        println(io, "  \"git_commit\": \"$(git_commit)\",")
        println(io, "  \"git_branch\": \"$(git_branch)\",")
        println(io, "  \"git_dirty\": $(git_dirty),")
    catch
    end
    println(io, "  \"seed\": $(Int(get(IDS_CFG, "seed", 0))),")
    println(io, "  \"make_gif\": $(get(IDS_CFG, "make_gif", false)),")
    println(io, "  \"gif_stride\": $(Int(get(IDS_CFG, "gif_stride", 0))),")
    println(io, "  \"rt_gif_stride\": $(Int(get(IDS_CFG, "rt_gif_stride", 0))),")
    println(io, "  \"output_dir\": \"$(get(IDS_CFG, "output_dir", "output"))\"")
    println(io, "}")
end

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
    seed_env              = Int(get(IDS_CFG, "seed", 123))
    n = Int(get(IDS_CFG, "n", 300))

    _tstart("generate")
    log("[static] generating $(n) observations …")
    env = Environment(initial_state, observation_precision; seed=seed_env)
    for i in 1:n
        getnext!(env)
        if i % 50 == 0
            log("[static] generated $(i)/$(n)")
        end
    end
    _tstop("generate")

    history = gethistory(env)
    observations = getobservations(env)
    datastream = to_namedtuple_stream(observations)

    autoupdates = make_autoupdates()
    init = make_initialization()

    _tstart("static_infer")
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
    # Persist stepwise metrics for diagnostics and parity with realtime
    upto_s = min(length(μ), length(history), length(observations))
    r_s = history[1:upto_s] .- μ[1:upto_s]
    mae_s = abs.(r_s)
    mse_s = r_s .^ 2
    writedlm(joinpath(static_dir, "static_metrics_stepwise.csv"), hcat(collect(1:upto_s), μ[1:upto_s], σ2[1:upto_s], history[1:upto_s], observations[1:upto_s], r_s, mae_s, mse_s))

    _tstop("static_infer")
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
        _tstart("static_fe_prefix")
        fe_t = compute_per_timestep_fe(observations; iterations=10)
        # Persist numeric outputs (CSV) immediately to allow reuse later without recomputation
        writedlm(joinpath(static_dir, "static_free_energy.csv"), fe_t)
        anim_fe = InfiniteDataStreamViz.animate_free_energy(fe_t; stride=stride)
        static_fe_gif = joinpath(static_dir, "static_free_energy.gif")
        InfiniteDataStreamViz.save_gif(anim_fe, static_fe_gif)
        log("[static] saved: $(static_fe_gif)")
        # Composed animation (estimates + free energy)
        anim_comp = InfiniteDataStreamViz.animate_composed_estimates_fe(μ, σ2, history, observations, fe_t; stride=stride)
        static_comp_gif = joinpath(static_dir, "static_composed_estimates_fe.gif")
        InfiniteDataStreamViz.save_gif(anim_comp, static_comp_gif)
        _tstop("static_fe_prefix")
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
    _apply_safe_limits!(pfe, fe_t_plot)
    png(pfe, joinpath(static_dir, "static_free_energy.png"))
    writedlm(joinpath(static_dir, "static_posterior_x_current.csv"), hcat(μ, σ2))
    writedlm(joinpath(static_dir, "static_truth_history.csv"), history)
    writedlm(joinpath(static_dir, "static_observations.csv"), observations)
    # Stepwise metrics header (additional file for readability)
    open(joinpath(static_dir, "static_metrics_stepwise_with_header.csv"), "w") do io
        println(io, "t,mu,var,truth,obs,residual,abs_residual,residual2")
    end
    open(joinpath(static_dir, "static_metrics_stepwise_with_header.csv"), "a") do io
        writedlm(io, hcat(collect(1:upto_s), μ[1:upto_s], σ2[1:upto_s], history[1:upto_s], observations[1:upto_s], r_s, mae_s, mse_s), ',')
    end
    if tau_series !== nothing
        tau_shape = shape.(tau_series)
        tau_rate  = rate.(tau_series)
        tau_mean  = tau_shape ./ tau_rate
        writedlm(joinpath(static_dir, "static_posterior_tau_shape_rate.csv"), hcat(tau_shape, tau_rate))
        pτ = plot(tau_mean; label="E[τ]", xlabel="t", ylabel="precision")
        _apply_safe_limits!(pτ, tau_mean)
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
    seed_env              = Int(get(IDS_CFG, "seed", 123))
    n = Int(get(IDS_CFG, "n", 1000))  # Match static mode
    interval_ms = Int(get(IDS_CFG, "interval_ms", 10))  # Higher frequency: 100Hz instead of ~24Hz
    rt_iters = Int(get(IDS_CFG, "rt_iterations", get(IDS_CFG, "iterations", 50)))  # Much more iterations per step for better convergence

    env = Environment(initial_state, observation_precision; seed=seed_env)

    producer() = getnext!(env)
    observations = timer_observations(interval_ms, n, producer)
    datastream = observations |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d, ))

    autoupdates = make_autoupdates()
    init = make_initialization()

    _tstart("realtime_infer")
    log("[realtime] building engine with live FE stream …")
    engine = create_engine_with_fe_stream(
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = datastream,
        autoupdates    = autoupdates,
        returnvars     = (:x_current, ),
        initialization = init,
        iterations     = rt_iters,
        free_energy    = true,
        keephistory    = Int(get(IDS_CFG, "keephistory", 10_000)),
        historyvars    = (x_current = KeepLast(), τ = KeepLast()),
        autostart      = false,
    )

    # Collect streamed free energy if exposed and simple progress counter
    mu_rt = Float64[]
    var_rt = Float64[]
    fe_rt  = Float64[]
    tau_shape_rt = Float64[]
    tau_rate_rt  = Float64[]
    # For consistent summary accounting across inner scopes
    posterior_samples_written = 0
    fe_points_written = 0
    obs_count = Ref(0)
    # Strict FE per-step computation buffer (online), configurable frequency
    obs_buffer = Float64[]
    fe_rt_strict = Float64[]
    fe_every = try
        Int(get(IDS_CFG, "rt_fe_every", 1))  # Compute FE every observation for maximum tracking
    catch
        1
    end
    
    # Adaptive convergence parameters
    convergence_threshold = Float64(get(IDS_CFG, "convergence_threshold", 1e-4))  # Slightly relaxed for speed
    max_adaptive_iters = Int(get(IDS_CFG, "max_adaptive_iters", 100))  # Allow more iterations if needed
    min_adaptive_iters = Int(get(IDS_CFG, "min_adaptive_iters", rt_iters))
    # Count observations as they arrive to show progress
    _ = subscribe!(datastream, d -> begin
        obs_count[] += 1
        # Accumulate observations for strict online FE computation
        local yval
        try
            yval = hasproperty(d, :y) ? getfield(d, :y) : d
        catch
            yval = d
        end
        push!(obs_buffer, Float64(yval))
        if fe_every > 0 && (length(obs_buffer) % fe_every == 0)
            # Compute FE for current prefix using adaptive convergence
            try
                ds_t = Main.InfiniteDataStreamStreams.to_namedtuple_stream(obs_buffer)
                au = Main.InfiniteDataStreamUpdates.make_autoupdates()
                init = Main.InfiniteDataStreamUpdates.make_initialization()
                
                # Adaptive convergence: start with minimum iterations and increase if needed
                current_iters = min_adaptive_iters
                converged = false
                prev_fe = Inf
                
                while current_iters <= max_adaptive_iters && !converged
                    eng_t = infer(
                        model          = Main.InfiniteDataStreamModel.kalman_filter(),
                        constraints    = Main.InfiniteDataStreamModel.filter_constraints(),
                        datastream     = ds_t,
                        autoupdates    = au,
                        returnvars     = (:x_current,),
                        initialization = init,
                        iterations     = current_iters,
                        free_energy    = true,
                        keephistory    = 1,
                        autostart      = true,
                    )
                    
                    current_fe = eng_t.free_energy_history[end]
                    
                    # Check convergence
                    if abs(current_fe - prev_fe) < convergence_threshold
                        converged = true
                        push!(fe_rt_strict, current_fe)
                        break
                    end
                    
                    prev_fe = current_fe
                    current_iters += 5  # Increase iterations gradually
                end
                
                # If not converged, use the last computed value
                if !converged && current_iters > max_adaptive_iters
                    push!(fe_rt_strict, prev_fe)
                end
                
            catch e
                # Fallback to basic computation if adaptive fails
                try
                    eng_t = infer(
                        model          = Main.InfiniteDataStreamModel.kalman_filter(),
                        constraints    = Main.InfiniteDataStreamModel.filter_constraints(),
                        datastream     = ds_t,
                        autoupdates    = au,
                        returnvars     = (:x_current,),
                        initialization = init,
                        iterations     = rt_iters,
                        free_energy    = true,
                        keephistory    = 1,
                        autostart      = true,
                    )
                    push!(fe_rt_strict, eng_t.free_energy_history[end])
                catch
                end
            end
        end
    end)
    # Streamed free-energy (if exposed by engine)
    #
    # We try to subscribe to a live FE observable published by the engine. If available, we capture
    # the true realtime FE values. Otherwise, we compute a fallback FE series online (prefix-based)
    # using the realtime iteration budget. This mirrors the behavior described in REALTIME_IN_RXINFER.md.
    fe_stream_subscribed = Ref(false)
    try
        # Try common property names; RxInfer may expose as `free_energy`
        local fe_stream
        if hasproperty(engine, :free_energy)
            fe_stream = getproperty(engine, :free_energy)
        elseif hasproperty(engine, :free_energy_stream)
            fe_stream = getproperty(engine, :free_energy_stream)
        else
            fe_stream = nothing
        end
        if fe_stream !== nothing
            log("[realtime] subscribing to engine free-energy stream …")
            _ = subscribe!(fe_stream, fe -> begin
                try
                    push!(fe_rt, Float64(fe))
                    if length(fe_rt) % 50 == 0
                        log("[realtime] FE stream points captured=$(length(fe_rt)) …")
                    end
                catch
                end
            end)
            fe_stream_subscribed[] = true
        end
    catch e
        @warn "[realtime] failed to subscribe to FE stream" error=e
    end

    if fe_stream_subscribed[]
        log("[realtime] FE stream subscription active")
    else
        log("[realtime] FE stream not exposed pre-start; will fallback if empty")
    end
    log("[realtime] starting engine (interval=$(interval_ms)ms, n=$(n)) …")
    RxInfer.start(engine)
    # Some engines expose streams only after start; try late subscription
    if !fe_stream_subscribed[]
        try
            local fe_stream2 = hasproperty(engine, :free_energy) ? getproperty(engine, :free_energy) : (hasproperty(engine, :free_energy_stream) ? getproperty(engine, :free_energy_stream) : nothing)
            if fe_stream2 !== nothing
                log("[realtime] FE stream available post-start; subscribing …")
                _ = subscribe!(fe_stream2, fe -> try push!(fe_rt, Float64(fe)) catch; end)
                fe_stream_subscribed[] = true
            end
        catch
        end
    end
    # Wait for all observations to be processed with improved synchronization
    total = ceil(Int, n * interval_ms / 1000) + 15  # Much more buffer time for higher frequency and convergence
    fe_history_idx = Ref(0)
    local last_obs_count = 0
    local stable_count = 0
    
    for s in 1:total
        sleep(0.5)  # More frequent checks for higher frequency updates
        current_obs = obs_count[]
        
        if s % 4 == 0 || s == total  # Log every 2 seconds
            pct = round(100 * min(current_obs, n) / n; digits=1)
            fe_computed = length(fe_rt_strict)
            log("[realtime] elapsed $(s*0.5)s, observed=$(current_obs) / $(n) ($(pct)%), FE computed=$(fe_computed)")
        end
        
        # Check if observations have stabilized
        if current_obs == last_obs_count
            stable_count += 1
        else
            stable_count = 0
        end
        last_obs_count = current_obs
        
        # Continue until we have processed all n observations and they're stable
        if current_obs >= n && stable_count >= 4  # 2 seconds of stability
            log("[realtime] all $(n) observations processed and stable, waiting for final inference...")
            sleep(3)  # Allow final inference and FE computation to complete
            break
        end
    end
    # Persist artifacts

    # Save realtime posterior snapshot plot and CSVs
    # Prefer history snapshots for alignment with truth/obs
    let hist = gethistory(env), obs = getobservations(env)
        # If engine kept history, use it; otherwise fall back to subscribed arrays
        if haskey(engine.history, :x_current)
            est_rt = engine.history[:x_current]
            mu_rt = mean.(est_rt)
            var_rt = var.(est_rt)
        end
        m = length(mu_rt)
        if m > 0
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
            # Do not mirror static FE artifacts; only persist realtime FE if exposed below
            # Persist realtime FE if captured
            # Track the FE time series we actually persisted for subsequent PNG plotting
            fe_series_used = Float64[]
            if !isempty(fe_rt)
                writedlm(joinpath(realtime_dir, "realtime_free_energy.csv"), fe_rt)
                fe_points_written = length(fe_rt)
                fe_series_used = fe_rt
                if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
                    anim_fe_rt = InfiniteDataStreamViz.animate_free_energy(fe_rt; stride=stride_rt)
                    InfiniteDataStreamViz.save_gif(anim_fe_rt, joinpath(realtime_dir, "realtime_free_energy.gif"))
                    log("[realtime] saved: $(joinpath(realtime_dir, "realtime_free_energy.gif"))")
                    # Two-panel composed realtime animation (estimates + FE)
                    anim_rt_comp = InfiniteDataStreamViz.animate_composed_estimates_fe(mu_rt[1:upto], var_rt[1:upto], hist, obs, fe_rt; stride=stride_rt)
                    InfiniteDataStreamViz.save_gif(anim_rt_comp, joinpath(realtime_dir, "realtime_composed_estimates_fe.gif"))
                    log("[realtime] saved: $(joinpath(realtime_dir, "realtime_composed_estimates_fe.gif"))")
                end
            else
                # No live FE stream and we do not fabricate: persist empty CSV and a placeholder PNG, skip GIFs
                log("[realtime] FE stream not exposed; realtime FE will be empty (no fallback computed)")
                open(joinpath(realtime_dir, "realtime_free_energy.csv"), "w") do io; end
                try
                    pfe_rt_empty = plot(title="Realtime FE stream not available", xaxis=false, yaxis=false)
                    png(pfe_rt_empty, joinpath(realtime_dir, "realtime_free_energy.png"))
                catch; end
            end
            # Save a static FE PNG comparable to the static mode
            if !isempty(fe_series_used)
                pfe_rt = plot(fe_series_used; label="Bethe Free Energy (averaged)")
                _apply_safe_limits!(pfe_rt, fe_series_used)
                png(pfe_rt, joinpath(realtime_dir, "realtime_free_energy.png"))
            end
            # Realtime composed GIFs
            if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
                log("[realtime] GIF generation enabled via IDS_MAKE_GIF=1")
            else
                log("[realtime] GIF generation disabled (set IDS_MAKE_GIF=1 to enable)")
            end
            writedlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"), hcat(mu_rt[1:upto], var_rt[1:upto]))
            # Stepwise metrics and tracing
            r_rt = hist[1:upto] .- mu_rt[1:upto]
            mae_rt = abs.(r_rt)
            mse_rt = r_rt .^ 2
            writedlm(joinpath(realtime_dir, "realtime_metrics_stepwise.csv"), hcat(collect(1:upto), mu_rt[1:upto], var_rt[1:upto], hist[1:upto], obs[1:upto], r_rt, mae_rt, mse_rt))
            open(joinpath(realtime_dir, "realtime_metrics_stepwise_with_header.csv"), "w") do io
                println(io, "t,mu,var,truth,obs,residual,abs_residual,residual2")
            end
            open(joinpath(realtime_dir, "realtime_metrics_stepwise_with_header.csv"), "a") do io
                writedlm(io, hcat(collect(1:upto), mu_rt[1:upto], var_rt[1:upto], hist[1:upto], obs[1:upto], r_rt, mae_rt, mse_rt), ',')
            end
            posterior_samples_written = upto
            # τ from history if present
            if haskey(engine.history, :τ)
                τ_hist = engine.history[:τ]
                tau_shape_rt = shape.(τ_hist)
                tau_rate_rt = rate.(τ_hist)
            end
            if !isempty(tau_shape_rt)
                writedlm(joinpath(realtime_dir, "realtime_posterior_tau_shape_rate.csv"), hcat(tau_shape_rt[1:upto], tau_rate_rt[1:upto]))
                tau_mean_rt = tau_shape_rt[1:upto] ./ tau_rate_rt[1:upto]
                pτrt = plot(tau_mean_rt; label="E[τ] realtime", xlabel="t", ylabel="precision")
                _apply_safe_limits!(pτrt, tau_mean_rt)
                png(pτrt, joinpath(realtime_dir, "realtime_tau_mean.png"))
            end
        end
    end
    # Persist truth/observations matched to posterior length to avoid misalignment
    if posterior_samples_written > 0
        hist_full = gethistory(env)
        obs_full = getobservations(env)
        upto_io = min(posterior_samples_written, length(hist_full), length(obs_full))
        writedlm(joinpath(realtime_dir, "realtime_truth_history.csv"), hist_full[1:upto_io])
        writedlm(joinpath(realtime_dir, "realtime_observations.csv"), obs_full[1:upto_io])
    else
        # Fallback if something unexpected happened
        writedlm(joinpath(realtime_dir, "realtime_truth_history.csv"), gethistory(env))
        writedlm(joinpath(realtime_dir, "realtime_observations.csv"), getobservations(env))
    end

    open(joinpath(realtime_dir, "realtime_summary.txt"), "w") do io
        println(io, "realtime_run_completed=true at ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "n=$(n), interval_ms=$(interval_ms)")
        println(io, "iterations=$(rt_iters)")
        println(io, "seed=$(seed_env)")
        println(io, "posterior_samples_captured=$(posterior_samples_written)")
        println(io, "fe_stream_subscribed=$(fe_stream_subscribed[])")
        println(io, "fe_stream_points=$(length(fe_rt))")
        println(io, "free_energy_points_captured=$(fe_points_written)")
        # Aggregate metrics if available
        try
            met_rt = readdlm(joinpath(realtime_dir, "realtime_metrics_stepwise.csv"))
            avg_mae = mean(met_rt[:,7])
            avg_mse = mean(met_rt[:,8])
            println(io, "mae_realtime_avg=$(avg_mae)")
            println(io, "mse_realtime_avg=$(avg_mse)")
        catch
        end
    end
    _tstop("realtime_infer")
    log("[realtime] done")
end

log("All outputs saved to $(outdir)")

# Persist timings if available
try
    open(joinpath(outdir, "timings.json"), "w") do io
        println(io, "{")
        first = true
        for (k,v) in _timings
            if !first; println(io, ","); end
            print(io, "  \"$(k)\": $(round(v; digits=3))")
            first = false
        end
        println(io)
        println(io, "}")
    end
catch
end

# Comparison report (errors and overlay plots)
try
    # Load static
    static_truth = readdlm(joinpath(static_dir, "static_truth_history.csv"))[:]
    static_est   = readdlm(joinpath(static_dir, "static_posterior_x_current.csv"))
    μs = static_est[:,1]
    σ2s = static_est[:,2]

    # Load realtime
    rt_truth = readdlm(joinpath(realtime_dir, "realtime_truth_history.csv"))[:]
    rt_est   = readdlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"))
    μr = rt_est[:,1]
    σ2r = rt_est[:,2]

    upto = min(length(static_truth), length(μs), length(rt_truth), length(μr))
    
    # Comprehensive error metrics
    residuals_static = μs[1:upto] .- static_truth[1:upto]
    residuals_rt = μr[1:upto] .- rt_truth[1:upto]
    
    # Basic metrics
    mse_static = mean(residuals_static.^2)
    mse_rt = mean(residuals_rt.^2)
    mae_static = mean(abs.(residuals_static))
    mae_rt = mean(abs.(residuals_rt))
    rmse_static = sqrt(mse_static)
    rmse_rt = sqrt(mse_rt)
    
    # Advanced statistical metrics
    using Statistics
    
    # Bias (systematic error)
    bias_static = mean(residuals_static)
    bias_rt = mean(residuals_rt)
    
    # Variance of residuals
    var_residuals_static = var(residuals_static)
    var_residuals_rt = var(residuals_rt)
    
    # Standard deviation of residuals
    std_residuals_static = std(residuals_static)
    std_residuals_rt = std(residuals_rt)
    
    # Correlation between predictions and truth
    cor_static = cor(μs[1:upto], static_truth[1:upto])
    cor_rt = cor(μr[1:upto], rt_truth[1:upto])
    
    # R-squared (coefficient of determination)
    ss_res_static = sum(residuals_static.^2)
    ss_tot_static = sum((static_truth[1:upto] .- mean(static_truth[1:upto])).^2)
    r2_static = 1 - ss_res_static / ss_tot_static
    
    ss_res_rt = sum(residuals_rt.^2)
    ss_tot_rt = sum((rt_truth[1:upto] .- mean(rt_truth[1:upto])).^2)
    r2_rt = 1 - ss_res_rt / ss_tot_rt
    
    # Quantile-based metrics
    q25_abs_err_static = quantile(abs.(residuals_static), 0.25)
    q50_abs_err_static = quantile(abs.(residuals_static), 0.50)  # median absolute error
    q75_abs_err_static = quantile(abs.(residuals_static), 0.75)
    q95_abs_err_static = quantile(abs.(residuals_static), 0.95)
    
    q25_abs_err_rt = quantile(abs.(residuals_rt), 0.25)
    q50_abs_err_rt = quantile(abs.(residuals_rt), 0.50)
    q75_abs_err_rt = quantile(abs.(residuals_rt), 0.75)
    q95_abs_err_rt = quantile(abs.(residuals_rt), 0.95)
    
    # Maximum absolute error
    max_abs_err_static = maximum(abs.(residuals_static))
    max_abs_err_rt = maximum(abs.(residuals_rt))
    
    # Normalized metrics (relative to truth range)
    truth_range_static = maximum(static_truth[1:upto]) - minimum(static_truth[1:upto])
    truth_range_rt = maximum(rt_truth[1:upto]) - minimum(rt_truth[1:upto])
    
    nrmse_static = rmse_static / truth_range_static
    nrmse_rt = rmse_rt / truth_range_rt
    nmae_static = mae_static / truth_range_static
    nmae_rt = mae_rt / truth_range_rt

    # Save comprehensive metrics
    open(joinpath(cmp_dir, "metrics.txt"), "w") do io
        println(io, "# Basic Error Metrics")
        println(io, "mse_static=", mse_static)
        println(io, "mse_realtime=", mse_rt)
        println(io, "mae_static=", mae_static)
        println(io, "mae_realtime=", mae_rt)
        println(io, "rmse_static=", rmse_static)
        println(io, "rmse_realtime=", rmse_rt)
        println(io, "")
        println(io, "# Bias and Variance")
        println(io, "bias_static=", bias_static)
        println(io, "bias_realtime=", bias_rt)
        println(io, "var_residuals_static=", var_residuals_static)
        println(io, "var_residuals_realtime=", var_residuals_rt)
        println(io, "std_residuals_static=", std_residuals_static)
        println(io, "std_residuals_realtime=", std_residuals_rt)
        println(io, "")
        println(io, "# Correlation and R-squared")
        println(io, "correlation_static=", cor_static)
        println(io, "correlation_realtime=", cor_rt)
        println(io, "r2_static=", r2_static)
        println(io, "r2_realtime=", r2_rt)
        println(io, "")
        println(io, "# Quantile-based Metrics")
        println(io, "q25_abs_err_static=", q25_abs_err_static)
        println(io, "q50_abs_err_static=", q50_abs_err_static)
        println(io, "q75_abs_err_static=", q75_abs_err_static)
        println(io, "q95_abs_err_static=", q95_abs_err_static)
        println(io, "max_abs_err_static=", max_abs_err_static)
        println(io, "")
        println(io, "q25_abs_err_realtime=", q25_abs_err_rt)
        println(io, "q50_abs_err_realtime=", q50_abs_err_rt)
        println(io, "q75_abs_err_realtime=", q75_abs_err_rt)
        println(io, "q95_abs_err_realtime=", q95_abs_err_rt)
        println(io, "max_abs_err_realtime=", max_abs_err_rt)
        println(io, "")
        println(io, "# Normalized Metrics")
        println(io, "nrmse_static=", nrmse_static)
        println(io, "nrmse_realtime=", nrmse_rt)
        println(io, "nmae_static=", nmae_static)
        println(io, "nmae_realtime=", nmae_rt)
        println(io, "")
        println(io, "# Sample Size")
        println(io, "n_compare=", upto)
        println(io, "matched_timesteps=", upto)
        println(io, "truth_range_static=", truth_range_static)
        println(io, "truth_range_realtime=", truth_range_rt)
    end

    # Write a JSON-like summary for downstream analysis
    try
        # FE stats if available
        fe_static_stats = "{}"; fe_rt_stats = "{}"
        try
            fe_s = readdlm(joinpath(static_dir, "static_free_energy.csv"))[:]
            if !isempty(fe_s)
                fe_static_stats = "{\"count\": $(length(fe_s)), \"median\": $(median(fe_s)), \"p90\": $(quantile(fe_s, 0.90))}"
            end
        catch; end
        try
            fe_r = readdlm(joinpath(realtime_dir, "realtime_free_energy.csv"))[:]
            if !isempty(fe_r)
                fe_rt_stats = "{\"count\": $(length(fe_r)), \"median\": $(median(fe_r)), \"p90\": $(quantile(fe_r, 0.90))}"
            end
        catch; end
        # Persist comprehensive summary
        open(joinpath(outdir, "summary.json"), "w") do io
            println(io, "{")
            println(io, "  \"sample_size\": {")
            println(io, "    \"n_compare\": $(upto),")
            println(io, "    \"matched_timesteps\": $(upto)")
            println(io, "  },")
            println(io, "  \"basic_metrics\": {")
            println(io, "    \"mae_static\": $(mae_static),")
            println(io, "    \"mae_realtime\": $(mae_rt),")
            println(io, "    \"mse_static\": $(mse_static),")
            println(io, "    \"mse_realtime\": $(mse_rt),")
            println(io, "    \"rmse_static\": $(rmse_static),")
            println(io, "    \"rmse_realtime\": $(rmse_rt)")
            println(io, "  },")
            println(io, "  \"bias_variance\": {")
            println(io, "    \"bias_static\": $(bias_static),")
            println(io, "    \"bias_realtime\": $(bias_rt),")
            println(io, "    \"var_residuals_static\": $(var_residuals_static),")
            println(io, "    \"var_residuals_realtime\": $(var_residuals_rt)")
            println(io, "  },")
            println(io, "  \"correlation\": {")
            println(io, "    \"correlation_static\": $(cor_static),")
            println(io, "    \"correlation_realtime\": $(cor_rt),")
            println(io, "    \"r2_static\": $(r2_static),")
            println(io, "    \"r2_realtime\": $(r2_rt)")
            println(io, "  },")
            println(io, "  \"quantile_metrics\": {")
            println(io, "    \"q50_abs_err_static\": $(q50_abs_err_static),")
            println(io, "    \"q50_abs_err_realtime\": $(q50_abs_err_rt),")
            println(io, "    \"q95_abs_err_static\": $(q95_abs_err_static),")
            println(io, "    \"q95_abs_err_realtime\": $(q95_abs_err_rt),")
            println(io, "    \"max_abs_err_static\": $(max_abs_err_static),")
            println(io, "    \"max_abs_err_realtime\": $(max_abs_err_rt)")
            println(io, "  },")
            println(io, "  \"normalized_metrics\": {")
            println(io, "    \"nrmse_static\": $(nrmse_static),")
            println(io, "    \"nrmse_realtime\": $(nrmse_rt),")
            println(io, "    \"nmae_static\": $(nmae_static),")
            println(io, "    \"nmae_realtime\": $(nmae_rt)")
            println(io, "  },")
            println(io, "  \"coverage\": {")
            println(io, "    \"coverage_static_95\": $(coverage_static),")
            println(io, "    \"coverage_realtime_95\": $(coverage_rt)")
            println(io, "  },")
            println(io, "  \"fe_stats\": {")
            println(io, "    \"fe_static\": $(fe_static_stats),")
            println(io, "    \"fe_realtime\": $(fe_rt_stats)")
            println(io, "  }")
            println(io, "}")
        end
    catch
    end

    # Overlay plot
    pcmp = plot(static_truth[1:upto]; label="truth", color=:black)
    _apply_safe_limits!(pcmp, static_truth[1:upto])
    plot!(pcmp, μs[1:upto]; label="static μ", color=:blue)
    plot!(pcmp, μr[1:upto]; label="realtime μ", color=:orange)
    png(pcmp, joinpath(cmp_dir, "means_compare.png"))

    # Comprehensive visualizations
    
    # 1) Scatter: static vs realtime means with statistics
    pscatter = scatter(μs[1:upto], μr[1:upto]; ms=3, alpha=0.7, label="points", xlabel="static μ", ylabel="realtime μ")
    plot!(pscatter, [minimum(μs[1:upto]); maximum(μs[1:upto])], [minimum(μs[1:upto]); maximum(μs[1:upto])]; label="y=x", color=:gray, lw=1.5)
    # Add correlation info to title
    title!(pscatter, "Static vs Realtime Predictions (r=$(round(cor(μs[1:upto], μr[1:upto]); digits=3)))")
    png(pscatter, joinpath(cmp_dir, "scatter_static_vs_realtime.png"))

    # 2) Residuals analysis with enhanced plots
    r_static = residuals_static
    r_rt = residuals_rt
    
    # Individual residual plots
    pr_s = plot(r_static; label="residual (static)", xlabel="t", ylabel="truth - mean", size=(1000,300), color=:blue)
    _apply_safe_limits!(pr_s, r_static)
    hline!(pr_s, [0.0]; color=:gray, lw=1, label="zero line")
    hline!(pr_s, [bias_static]; color=:red, lw=1, linestyle=:dash, label="bias=$(round(bias_static; digits=3))")
    title!(pr_s, "Static Residuals (RMSE=$(round(rmse_static; digits=3)))")
    png(pr_s, joinpath(cmp_dir, "residuals_static.png"))
    
    pr_r = plot(r_rt; label="residual (realtime)", xlabel="t", ylabel="truth - mean", size=(1000,300), color=:orange)
    _apply_safe_limits!(pr_r, r_rt)
    hline!(pr_r, [0.0]; color=:gray, lw=1, label="zero line")
    hline!(pr_r, [bias_rt]; color=:red, lw=1, linestyle=:dash, label="bias=$(round(bias_rt; digits=3))")
    title!(pr_r, "Realtime Residuals (RMSE=$(round(rmse_rt; digits=3)))")
    png(pr_r, joinpath(cmp_dir, "residuals_realtime.png"))
    
    # Combined residuals plot
    pr_combined = plot(r_static; label="static residuals", xlabel="t", ylabel="residual", size=(1200,400), color=:blue, alpha=0.7)
    _apply_safe_limits!(pr_combined, vcat(r_static, r_rt))
    plot!(pr_combined, r_rt; label="realtime residuals", color=:orange, alpha=0.7)
    hline!(pr_combined, [0.0]; color=:gray, lw=2, label="zero line")
    title!(pr_combined, "Residuals Comparison")
    png(pr_combined, joinpath(cmp_dir, "residuals_combined.png"))
    
    # 3) Box plots for error distribution comparison
    using StatsPlots
    abs_err_static = abs.(r_static)
    abs_err_rt = abs.(r_rt)
    
    pbox = boxplot(["Static"], abs_err_static; label="", color=:blue, alpha=0.7)
    boxplot!(pbox, ["Realtime"], abs_err_rt; label="", color=:orange, alpha=0.7)
    ylabel!(pbox, "Absolute Error")
    title!(pbox, "Error Distribution Comparison")
    png(pbox, joinpath(cmp_dir, "error_boxplot.png"))
    
    # 4) Histogram of errors
    phist = histogram(abs_err_static; bins=30, alpha=0.7, label="static |error|", color=:blue, normalize=:probability)
    histogram!(phist, abs_err_rt; bins=30, alpha=0.7, label="realtime |error|", color=:orange, normalize=:probability)
    xlabel!(phist, "Absolute Error")
    ylabel!(phist, "Probability Density")
    title!(phist, "Error Distribution")
    png(phist, joinpath(cmp_dir, "error_histogram.png"))
    
    # 5) Q-Q plot for error distribution comparison
    using Plots
    sorted_static = sort(abs_err_static)
    sorted_rt = sort(abs_err_rt)
    n_min = min(length(sorted_static), length(sorted_rt))
    quantiles = (1:n_min) ./ n_min
    
    pqq = scatter(sorted_static[1:n_min], sorted_rt[1:n_min]; ms=2, alpha=0.7, label="", xlabel="Static |Error| Quantiles", ylabel="Realtime |Error| Quantiles")
    plot!(pqq, [0; maximum([sorted_static[n_min], sorted_rt[n_min]])], [0; maximum([sorted_static[n_min], sorted_rt[n_min]])]; color=:gray, lw=1.5, label="y=x")
    title!(pqq, "Q-Q Plot: Error Distributions")
    png(pqq, joinpath(cmp_dir, "qq_plot_errors.png"))
    
    # 6) Uncertainty comparison (using variance estimates)
    punc = plot(σ2s[1:upto]; label="static variance", xlabel="t", ylabel="variance", color=:blue, alpha=0.7)
    _apply_safe_limits!(punc, vcat(σ2s[1:upto], σ2r[1:upto]))
    plot!(punc, σ2r[1:upto]; label="realtime variance", color=:orange, alpha=0.7)
    title!(punc, "Uncertainty Estimates Comparison")
    png(punc, joinpath(cmp_dir, "uncertainty_comparison.png"))
    
    # 7) Prediction intervals comparison
    # 95% prediction intervals
    ci_static_lower = μs[1:upto] .- 1.96 .* sqrt.(σ2s[1:upto])
    ci_static_upper = μs[1:upto] .+ 1.96 .* sqrt.(σ2s[1:upto])
    ci_rt_lower = μr[1:upto] .- 1.96 .* sqrt.(σ2r[1:upto])
    ci_rt_upper = μr[1:upto] .+ 1.96 .* sqrt.(σ2r[1:upto])
    
    # Coverage analysis
    coverage_static = mean((static_truth[1:upto] .>= ci_static_lower) .& (static_truth[1:upto] .<= ci_static_upper))
    coverage_rt = mean((rt_truth[1:upto] .>= ci_rt_lower) .& (rt_truth[1:upto] .<= ci_rt_upper))
    
    pci = plot(static_truth[1:upto]; label="truth", color=:black, lw=2)
    _apply_safe_limits!(pci, static_truth[1:upto])
    plot!(pci, μs[1:upto]; ribbon=(μs[1:upto] .- ci_static_lower, ci_static_upper .- μs[1:upto]), 
          label="static (coverage=$(round(coverage_static*100; digits=1))%)", alpha=0.3, color=:blue)
    plot!(pci, μr[1:upto]; ribbon=(μr[1:upto] .- ci_rt_lower, ci_rt_upper .- μr[1:upto]), 
          label="realtime (coverage=$(round(coverage_rt*100; digits=1))%)", alpha=0.3, color=:orange)
    xlabel!(pci, "t")
    ylabel!(pci, "value")
    title!(pci, "95% Prediction Intervals")
    png(pci, joinpath(cmp_dir, "prediction_intervals.png"))
    
    # 8) Running metrics over time
    window_size = max(10, upto ÷ 20)  # Adaptive window size
    running_mae_static = Float64[]
    running_mae_rt = Float64[]
    running_rmse_static = Float64[]
    running_rmse_rt = Float64[]
    
    for i in window_size:upto
        start_idx = max(1, i - window_size + 1)
        push!(running_mae_static, mean(abs.(r_static[start_idx:i])))
        push!(running_mae_rt, mean(abs.(r_rt[start_idx:i])))
        push!(running_rmse_static, sqrt(mean(r_static[start_idx:i].^2)))
        push!(running_rmse_rt, sqrt(mean(r_rt[start_idx:i].^2)))
    end
    
    prun = plot(window_size:upto, running_mae_static; label="static MAE", color=:blue, lw=2)
    _apply_safe_limits!(prun, vcat(running_mae_static, running_mae_rt))
    plot!(prun, window_size:upto, running_mae_rt; label="realtime MAE", color=:orange, lw=2)
    xlabel!(prun, "t")
    ylabel!(prun, "Running MAE")
    title!(prun, "Running Mean Absolute Error (window=$(window_size))")
    png(prun, joinpath(cmp_dir, "running_mae.png"))
    
    prun_rmse = plot(window_size:upto, running_rmse_static; label="static RMSE", color=:blue, lw=2)
    _apply_safe_limits!(prun_rmse, vcat(running_rmse_static, running_rmse_rt))
    plot!(prun_rmse, window_size:upto, running_rmse_rt; label="realtime RMSE", color=:orange, lw=2)
    xlabel!(prun_rmse, "t")
    ylabel!(prun_rmse, "Running RMSE")
    title!(prun_rmse, "Running Root Mean Square Error (window=$(window_size))")
    png(prun_rmse, joinpath(cmp_dir, "running_rmse.png"))

    # Optional comparison animation (overlay evolving)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        stride_cmp = parse(Int, get(ENV, "IDS_GIF_STRIDE", "5"))
        anim_overlay = InfiniteDataStreamViz.animate_overlay_means(static_truth, μs, μr; stride=stride_cmp)
        InfiniteDataStreamViz.save_gif(anim_overlay, joinpath(cmp_dir, "overlay_means.gif"))
    else
        @info "[comparison] overlay animation disabled (set IDS_MAKE_GIF=1 to enable)"
    end

    # Free energy comparison (if both available)
    try
        fe_static = readdlm(joinpath(static_dir, "static_free_energy.csv"))[:]
        fe_rt_cmp = readdlm(joinpath(realtime_dir, "realtime_free_energy.csv"))[:]
        upto_fe = min(length(fe_static), length(fe_rt_cmp))
        pfe_cmp = InfiniteDataStreamViz.plot_fe_comparison(fe_static[1:upto_fe], fe_rt_cmp[1:upto_fe])
        _apply_safe_limits!(pfe_cmp, vcat(fe_static[1:upto_fe], fe_rt_cmp[1:upto_fe]))
        png(pfe_cmp, joinpath(cmp_dir, "free_energy_compare.png"))
        # Scatter plot of free energies through time for both modes
        try
            t_fe = collect(1:upto_fe)
            pfe_sc = scatter(t_fe, fe_static[1:upto_fe]; label = "static FE", ms = 3, alpha = 0.7, xlabel = "t", ylabel = "free energy")
            scatter!(pfe_sc, t_fe, fe_rt_cmp[1:upto_fe]; label = "realtime FE", ms = 3, alpha = 0.7)
            png(pfe_sc, joinpath(cmp_dir, "free_energy_scatter.png"))
        catch
        end
        if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
            stride_cmp = parse(Int, get(ENV, "IDS_GIF_STRIDE", "5"))
            anim_cmp = InfiniteDataStreamViz.animate_comparison_static_vs_realtime(static_truth, μs, σ2s, μr, σ2r, fe_static, fe_rt_cmp; stride=stride_cmp)
            InfiniteDataStreamViz.save_gif(anim_cmp, joinpath(cmp_dir, "static_vs_realtime_composed.gif"))
        end
    catch
        @info "[comparison] FE series missing; skipping FE comparison"
    end

    # τ comparison if both present (τ is observation precision, i.e., inverse variance)
    try
        tau_sr = readdlm(joinpath(static_dir, "static_posterior_tau_shape_rate.csv"))
        tau_rr = readdlm(joinpath(realtime_dir, "realtime_posterior_tau_shape_rate.csv"))
        tau_mean_s = tau_sr[:,1] ./ tau_sr[:,2]
        tau_mean_r = tau_rr[:,1] ./ tau_rr[:,2]
        upto_tau = min(length(tau_mean_s), length(tau_mean_r))
        pτcmp = InfiniteDataStreamViz.plot_tau_comparison(tau_mean_s[1:upto_tau], tau_mean_r[1:upto_tau])
        png(pτcmp, joinpath(cmp_dir, "tau_comparison.png"))
    catch
        @info "[comparison] τ series missing; skipping tau comparison"
    end
catch e
    @warn "comparison report failed" error=e
end

# Clean up engine wrapper if used in realtime mode
try
    if @isdefined(engine) && hasproperty(engine, :_is_started)
        InfiniteDataStreamEngineWrapper.stop!(engine)
        log("Engine wrapper cleaned up")
    end
catch e
    @warn "Error cleaning up engine wrapper" error=e
end

