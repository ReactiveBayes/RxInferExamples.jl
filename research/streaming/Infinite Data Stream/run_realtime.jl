include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!(); const IDS_CFG = InfiniteDataStreamUtils.load_config()
using Dates, DelimitedFiles, Plots, RxInfer
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

# Helper: compute per-timestep FE by re-inferring on growing prefixes (no mirroring)
function compute_per_timestep_fe(observations::Vector{Float64}; iterations::Int=10)
    fe_series = Float64[]
    total = length(observations)
    log_stride = try
        Int(get(IDS_CFG, "fe_log_stride", 10))
    catch
        10
    end
    start_t = time()
    log("[realtime] FE prefix re-inference: total=$(total), iters/step=$(iterations), log_stride=$(log_stride)")
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
            log("[realtime] FE computed for t=$(t)/$(total) ($(pct)%); ETA ≈ $(eta_s)s")
        end
    end
    return fe_series
end

mkpath(get(IDS_CFG, "output_dir", "output"))
ts = Dates.format(now(), "yyyymmdd_HHMMSS")
outdir = joinpath(get(IDS_CFG, "output_dir", "output"), ts)
realtime_dir = joinpath(outdir, "realtime"); mkpath(realtime_dir)
log("Writing realtime artifacts under: $(outdir)")

initial_state         = Float64(get(IDS_CFG, "initial_state", 0.0))
observation_precision = Float64(get(IDS_CFG, "observation_precision", 0.1))
n = Int(get(IDS_CFG, "n", 300))
interval_ms = Int(get(IDS_CFG, "interval_ms", 41))
rt_iters = Int(get(IDS_CFG, "rt_iterations", get(IDS_CFG, "iterations", 10)))
seed_env = Int(get(IDS_CFG, "seed", 123))

env = Environment(initial_state, observation_precision; seed=seed_env)

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
    iterations     = rt_iters,
    free_energy    = true,
    keephistory    = Int(get(IDS_CFG, "keephistory", 10_000)),
    historyvars    = (x_current = KeepLast(), τ = KeepLast()),
    autostart      = false,
)

# Subscriptions for FE stream and progress
fe_rt  = Float64[]
fe_idx = Ref(0)
obs_count = Ref(0)
_ = subscribe!(datastream, _ -> (obs_count[] += 1))
try
    if hasproperty(engine, :free_energy) && engine.free_energy !== nothing
        _ = subscribe!(engine.free_energy, fe -> push!(fe_rt, Float64(fe)))
    end
catch
end

log("[realtime] starting engine (interval=$(interval_ms)ms, n=$(n)) …")
RxInfer.start(engine)

# Wait for completion with simple progress
total = ceil(Int, n * interval_ms / 1000) + 1
for s in 1:total
    sleep(1)
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
    if s % 2 == 0 || s == total
        pct = round(100 * min(obs_count[], n) / n; digits=1)
        log("[realtime] elapsed $(s)/$(total)s, observed=$(obs_count[]) / $(n) ($(pct)%)")
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

# Persist realtime FE if captured
if !isempty(fe_rt)
    writedlm(joinpath(realtime_dir, "realtime_free_energy.csv"), fe_rt)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        stride_rt = Int(get(IDS_CFG, "rt_gif_stride", 5))
        anim_fe_rt = InfiniteDataStreamViz.animate_free_energy(fe_rt; stride=stride_rt)
        InfiniteDataStreamViz.save_gif(anim_fe_rt, joinpath(realtime_dir, "realtime_free_energy.gif"))
    end
else
    # No FE stream exposed: compute FE offline from the realtime observations (still realtime-only)
    fe_rt_offline = compute_per_timestep_fe(obs; iterations=Int(get(IDS_CFG, "iterations", 10)))
    writedlm(joinpath(realtime_dir, "realtime_free_energy.csv"), fe_rt_offline)
    if get(ENV, "IDS_MAKE_GIF", get(IDS_CFG, "make_gif", true) ? "1" : "0") == "1"
        stride_rt = Int(get(IDS_CFG, "rt_gif_stride", 5))
        anim_fe_rt = InfiniteDataStreamViz.animate_free_energy(fe_rt_offline; stride=stride_rt)
        InfiniteDataStreamViz.save_gif(anim_fe_rt, joinpath(realtime_dir, "realtime_free_energy.gif"))
    end
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

log("[realtime] done; artifacts in $(realtime_dir)")

