module InfiniteDataStreamRunner

using Dates, DelimitedFiles, Plots, RxInfer

# NOTE: This runner expects the host process to have loaded modules via `include("utils.jl"); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()`
# We reference the already-loaded modules from `Main` for composability.
const InfiniteDataStreamModel = Main.InfiniteDataStreamModel
const InfiniteDataStreamEnv = Main.InfiniteDataStreamEnv
const InfiniteDataStreamUpdates = Main.InfiniteDataStreamUpdates
const InfiniteDataStreamStreams = Main.InfiniteDataStreamStreams
const InfiniteDataStreamViz = Main.InfiniteDataStreamViz

export RunArtifacts, run_static, run_realtime, compare_runs

struct RunArtifacts
    outdir::String
    n::Int
    mu::Vector{Float64}
    var::Vector{Float64}
    truth::Vector{Float64}
    obs::Vector{Float64}
    fe::Vector{Float64}
end

log(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)

function run_static(cfg::Dict{String,Any})::RunArtifacts
    initial_state         = Float64(get(cfg, "initial_state", 0.0))
    observation_precision = Float64(get(cfg, "observation_precision", 0.1))
    seed_env              = Int(get(cfg, "seed", 123))
    n = Int(get(cfg, "n", 300))

    env = InfiniteDataStreamEnv.Environment(initial_state, observation_precision; seed=seed_env)
    for _ in 1:n
        InfiniteDataStreamEnv.getnext!(env)
    end

    history = InfiniteDataStreamEnv.gethistory(env)
    observations = InfiniteDataStreamEnv.getobservations(env)
    ds = InfiniteDataStreamStreams.to_namedtuple_stream(observations)
    au = InfiniteDataStreamUpdates.make_autoupdates()
    init = InfiniteDataStreamUpdates.make_initialization()
    eng = infer(
        model          = InfiniteDataStreamModel.kalman_filter(),
        constraints    = InfiniteDataStreamModel.filter_constraints(),
        datastream     = ds,
        autoupdates    = au,
        returnvars     = (:x_current,),
        keephistory    = Int(get(cfg, "keephistory", 10_000)),
        historyvars    = (x_current = KeepLast(), τ = KeepLast()),
        initialization = init,
        iterations     = Int(get(cfg, "iterations", 10)),
        free_energy    = true,
        autostart      = true,
    )
    est = eng.history[:x_current]
    mu = mean.(est)
    va = var.(est)
    fe = begin
        fe_series = Float64[]
        for t in 1:length(observations)
            ds_t = InfiniteDataStreamStreams.to_namedtuple_stream(observations[1:t])
            eng_t = infer(
                model = InfiniteDataStreamModel.kalman_filter(), constraints = InfiniteDataStreamModel.filter_constraints(),
                datastream = ds_t, autoupdates = au, returnvars = (:x_current,),
                initialization = init, iterations = Int(get(cfg, "iterations", 10)),
                free_energy = true, keephistory=1, autostart=true,
            )
            push!(fe_series, eng_t.free_energy_history[end])
        end
        fe_series
    end
    outroot = get(cfg, "output_dir", "output")
    mkpath(outroot)
    ts = get(cfg, "ts", Dates.format(now(), "yyyymmdd_HHMMSS"))
    outdir = joinpath(outroot, ts, "static")
    mkpath(outdir)
    p = InfiniteDataStreamViz.plot_estimates(mu, va, history, observations; upto=n)
    png(p, joinpath(outdir, "static_inference.png"))
    writedlm(joinpath(outdir, "static_posterior_x_current.csv"), hcat(mu, va))
    writedlm(joinpath(outdir, "static_truth_history.csv"), history)
    writedlm(joinpath(outdir, "static_observations.csv"), observations)
    writedlm(joinpath(outdir, "static_free_energy.csv"), fe)
    return RunArtifacts(dirname(outdir), n, mu, va, history, observations, fe)
end

function run_realtime(cfg::Dict{String,Any})::RunArtifacts
    initial_state         = Float64(get(cfg, "initial_state", 0.0))
    observation_precision = Float64(get(cfg, "observation_precision", 0.1))
    seed_env              = Int(get(cfg, "seed", 123))
    n = Int(get(cfg, "n", 300))
    interval_ms = Int(get(cfg, "interval_ms", 41))
    rt_iters = Int(get(cfg, "rt_iterations", get(cfg, "iterations", 10)))
    rt_fe_every = Int(get(cfg, "rt_fe_every", 1))

    env = InfiniteDataStreamEnv.Environment(initial_state, observation_precision; seed=seed_env)
    producer() = InfiniteDataStreamEnv.getnext!(env)
    observations = InfiniteDataStreamStreams.timer_observations(interval_ms, n, producer)
    datastream = observations |> map(NamedTuple{(:y,), Tuple{Float64}}, d -> (y = d, ))
    au = InfiniteDataStreamUpdates.make_autoupdates()
    init = InfiniteDataStreamUpdates.make_initialization()
    eng = infer(
        model = InfiniteDataStreamModel.kalman_filter(), constraints = InfiniteDataStreamModel.filter_constraints(),
        datastream = datastream, autoupdates = au, returnvars = (:x_current,),
        initialization = init, iterations = rt_iters, free_energy = true,
        keephistory = Int(get(cfg, "keephistory", 10_000)), historyvars=(x_current=KeepLast(), τ=KeepLast()),
        autostart=false,
    )

    # Strict online FE
    obs_buf = Float64[]
    fe_strict = Float64[]
    _ = Rocket.subscribe!(datastream, d -> begin
        y = hasproperty(d,:y) ? getfield(d,:y) : d
        push!(obs_buf, Float64(y))
        if rt_fe_every > 0 && (length(obs_buf) % rt_fe_every == 0)
            try
                ds_t = InfiniteDataStreamStreams.to_namedtuple_stream(obs_buf)
                eng_t = infer(model = InfiniteDataStreamModel.kalman_filter(), constraints = InfiniteDataStreamModel.filter_constraints(),
                    datastream = ds_t, autoupdates = au, returnvars = (:x_current,),
                    initialization = init, iterations = rt_iters, free_energy=true,
                    keephistory=1, autostart=true)
                push!(fe_strict, eng_t.free_energy_history[end])
            catch
            end
        end
    end)

    RxInfer.start(eng)
    sleep(max(1, ceil(Int, n * interval_ms / 1000)))

    hist = InfiniteDataStreamEnv.gethistory(env)
    obs  = InfiniteDataStreamEnv.getobservations(env)
    mu = Float64[]; va = Float64[]
    if haskey(eng.history, :x_current)
        est_rt = eng.history[:x_current]
        mu = mean.(est_rt)
        va = var.(est_rt)
    end
    upto = min(length(mu), length(hist), length(obs))
    mu = mu[1:upto]; va = va[1:upto]
    fe = isempty(fe_strict) ? zeros(upto) : fe_strict[1:upto]
    outroot = get(cfg, "output_dir", "output")
    mkpath(outroot)
    ts = get(cfg, "ts", Dates.format(now(), "yyyymmdd_HHMMSS"))
    outdir = joinpath(outroot, ts, "realtime")
    mkpath(outdir)
    p = InfiniteDataStreamViz.plot_estimates(mu, va, hist, obs; upto=upto)
    png(p, joinpath(outdir, "realtime_inference.png"))
    writedlm(joinpath(outdir, "realtime_posterior_x_current.csv"), hcat(mu, va))
    writedlm(joinpath(outdir, "realtime_truth_history.csv"), hist[1:upto])
    writedlm(joinpath(outdir, "realtime_observations.csv"), obs[1:upto])
    writedlm(joinpath(outdir, "realtime_free_energy.csv"), fe)
    return RunArtifacts(dirname(outdir), upto, mu, va, hist[1:upto], obs[1:upto], fe)
end

function compare_runs(static::RunArtifacts, realtime::RunArtifacts; outdir::AbstractString="")
    out = isempty(outdir) ? joinpath(static.outdir, "comparison") : outdir
    mkpath(out)
    upto = min(length(static.truth), length(static.mu), length(realtime.truth), length(realtime.mu))
    μs = static.mu[1:upto]; μr = realtime.mu[1:upto]
    ts = static.truth[1:upto]; tr = realtime.truth[1:upto]
    mse_s = mean((μs .- ts).^2); mse_r = mean((μr .- tr).^2)
    mae_s = mean(abs.(μs .- ts)); mae_r = mean(abs.(μr .- tr))
    open(joinpath(out, "metrics.txt"), "w") do io
        println(io, "mse_static=", mse_s)
        println(io, "mse_realtime=", mse_r)
        println(io, "mae_static=", mae_s)
        println(io, "mae_realtime=", mae_r)
        println(io, "n_compare=", upto)
    end
    pcmp = plot(ts; label="truth", color=:black)
    plot!(pcmp, μs; label="static μ", color=:blue)
    plot!(pcmp, μr; label="realtime μ", color=:orange)
    png(pcmp, joinpath(out, "means_compare.png"))
    # FE side-by-side if both present
    if !isempty(static.fe) && !isempty(realtime.fe)
        upf = min(length(static.fe), length(realtime.fe))
        pfe = InfiniteDataStreamViz.plot_fe_comparison(static.fe[1:upf], realtime.fe[1:upf])
        png(pfe, joinpath(out, "free_energy_compare.png"))
    end
    # JSON-like summary
    try
        open(joinpath(out, "summary.json"), "w") do io
            println(io, "{")
            println(io, "  \"n_compare\": $(upto),")
            println(io, "  \"mae_static\": $(mae_s),")
            println(io, "  \"mse_static\": $(mse_s),")
            println(io, "  \"mae_realtime\": $(mae_r),")
            println(io, "  \"mse_realtime\": $(mse_r)")
            println(io, "}")
        end
    catch
    end
    return nothing
end

end # module


