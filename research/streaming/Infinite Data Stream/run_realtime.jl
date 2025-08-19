try
    @eval begin
        import Pkg
        Pkg.activate(@__DIR__)
        Pkg.instantiate()
    end
catch
end

include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!(); const IDS_CFG = InfiniteDataStreamUtils.load_config()
include("engine_wrapper.jl"); using .InfiniteDataStreamEngineWrapper
using Dates, DelimitedFiles, Plots, RxInfer, Rocket
using .InfiniteDataStreamModel
using .InfiniteDataStreamEnv
using .InfiniteDataStreamUpdates
using .InfiniteDataStreamStreams
using .InfiniteDataStreamViz

# Headless GR
if get(ENV, "GKSwstype", nothing) === nothing
    ENV["GKSwstype"] = "100"
end

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)

# No FE fabrication in realtime: we only record a live engine FE stream when exposed

mkpath(get(IDS_CFG, "output_dir", "output"))
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath(get(IDS_CFG, "output_dir", "output"), ts)
realtime_dir = joinpath(outdir, "realtime"); mkpath(realtime_dir)
log("Writing realtime artifacts under: $(outdir)")

initial_state         = Float64(get(IDS_CFG, "initial_state", 0.0))
observation_precision = Float64(get(IDS_CFG, "observation_precision", 0.1))
n = Int(get(IDS_CFG, "n", 1000))  # Match static mode
interval_ms = Int(get(IDS_CFG, "interval_ms", 10))  # Higher frequency: 100Hz
rt_iters = Int(get(IDS_CFG, "rt_iterations", get(IDS_CFG, "iterations", 50)))  # Much more iterations for better convergence
seed_env = Int(get(IDS_CFG, "seed", 123))

env = Environment(initial_state, observation_precision; seed=seed_env)

producer() = getnext!(env)
observations = timer_observations(interval_ms, n, producer)
datastream = observations |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d, ))

autoupdates = make_autoupdates()
init = make_initialization()

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

# Subscriptions for FE stream and progress
fe_rt = Float64[]
obs_count = Ref(0)
_ = subscribe!(datastream, _ -> (obs_count[] += 1))

# Subscribe to the live FE stream from our wrapper
fe_stream_subscribed = Ref(false)
try
    if hasproperty(engine, :free_energy) && engine.free_energy !== nothing
        _ = subscribe!(engine.free_energy, fe -> begin
            try
                push!(fe_rt, Float64(fe))
                if length(fe_rt) % 25 == 0
                    log("[realtime] FE stream captured $(length(fe_rt)) points …")
                end
            catch e
                @warn "Error capturing FE value" error=e
            end
        end)
        fe_stream_subscribed[] = true
        log("[realtime] subscribed to live FE stream from engine wrapper …")
    else
        log("[realtime] FE stream not available from wrapper")
    end
catch e
    @warn "[realtime] failed to subscribe to FE stream" error=e
end

log("[realtime] starting engine (interval=$(interval_ms)ms, n=$(n)) …")
RxInfer.start(engine)

# Wait for completion with improved synchronization
total = ceil(Int, n * interval_ms / 1000) + 15  # Much more buffer time for higher frequency and convergence
local last_obs_count = 0
local stable_count = 0

for s in 1:total
    sleep(0.5)  # More frequent checks
    current_obs = obs_count[]
    
    # opportunistically collect any new FE history entries if available
    try
        if hasproperty(engine, :free_energy_history) && engine.free_energy_history !== nothing
            fehist = engine.free_energy_history
            if length(fehist) > fe_idx[]
                append!(fe_rt, Float64.(fehist[fe_idx[]+1:end]))
                fe_idx[] = length(fehist)
            end
        end
    catch
    end
    
    if s % 4 == 0 || s == total  # Log every 2 seconds
        pct = round(100 * min(current_obs, n) / n; digits=1)
        fe_captured = length(fe_rt)
        log("[realtime] elapsed $(s*0.5)s, observed=$(current_obs) / $(n) ($(pct)%), FE captured=$(fe_captured)")
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
        sleep(3)  # Allow final inference to complete
        break
    end
end

# Persist artifacts
hist = gethistory(env)
obs  = getobservations(env)
if haskey(engine.history, :x_current)
    est_rt = engine.history[:x_current]
    mu_rt = mean.(est_rt)
    var_rt = var.(est_rt)
    upto = min(length(mu_rt), length(hist), length(obs))
    p_rt = InfiniteDataStreamViz.plot_estimates(mu_rt[1:upto], var_rt[1:upto], hist, obs; upto=upto)
    png(p_rt, joinpath(realtime_dir, "realtime_inference.png"))
    # GIF
    stride_rt = Int(get(IDS_CFG, "rt_gif_stride", 5))
    anim_rt = @animate for i in 1:stride_rt:upto
        InfiniteDataStreamViz.plot_estimates(mu_rt[1:upto], var_rt[1:upto], hist, obs; upto=i)
    end
    InfiniteDataStreamViz.save_gif(anim_rt, joinpath(realtime_dir, "realtime_inference.gif"))
    # CSVs
    writedlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"), hcat(mu_rt[1:upto], var_rt[1:upto]))
    # τ series if present
    if haskey(engine.history, :τ)
        τ_hist = engine.history[:τ]
        tau_shape_rt = shape.(τ_hist)
        tau_rate_rt = rate.(τ_hist)
        writedlm(joinpath(realtime_dir, "realtime_posterior_tau_shape_rate.csv"), hcat(tau_shape_rt[1:upto], tau_rate_rt[1:upto]))
    end
end

# Persist realtime FE strictly from live stream if present; otherwise write empty and a placeholder
if !isempty(fe_rt)
    writedlm(joinpath(realtime_dir, "realtime_free_energy.csv"), fe_rt)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        stride_rt = Int(get(IDS_CFG, "rt_gif_stride", 5))
        anim_fe_rt = InfiniteDataStreamViz.animate_free_energy(fe_rt; stride=stride_rt)
        InfiniteDataStreamViz.save_gif(anim_fe_rt, joinpath(realtime_dir, "realtime_free_energy.gif"))
    end
else
    open(joinpath(realtime_dir, "realtime_free_energy.csv"), "w") do io; end
    try
        pfe_rt_empty = plot(title="Realtime FE stream not available", xaxis=false, yaxis=false)
        png(pfe_rt_empty, joinpath(realtime_dir, "realtime_free_energy.png"))
    catch; end
end

writedlm(joinpath(realtime_dir, "realtime_truth_history.csv"), hist)
writedlm(joinpath(realtime_dir, "realtime_observations.csv"), obs)

open(joinpath(realtime_dir, "realtime_summary.txt"), "w") do io
    println(io, "realtime_run_completed=true at ", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    println(io, "n=$(n), interval_ms=$(interval_ms)")
    println(io, "iterations=$(Int(get(IDS_CFG, "iterations", 10)))")
    println(io, "seed=$(seed_env)")
    println(io, "posterior_samples_captured=$(min(length(hist), length(obs)))")
    println(io, "free_energy_points_captured=$(length(fe_rt))")
end

# Clean up the engine wrapper
try
    InfiniteDataStreamEngineWrapper.stop!(engine)
    log("[realtime] engine wrapper stopped and cleaned up")
catch e
    @warn "Error stopping engine wrapper" error=e
end

log("[realtime] done; artifacts in $(realtime_dir)")

