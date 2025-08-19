module InfiniteDataStreamUtils
using TOML

export load_modules!
export load_config

function load_modules!()
    # Load implementation files into Main so they are available as Main.InfiniteDataStream*
    thisdir = @__DIR__
    Base.include(Main, joinpath(thisdir, "model.jl"))
    Base.include(Main, joinpath(thisdir, "environment.jl"))
    Base.include(Main, joinpath(thisdir, "updates.jl"))
    Base.include(Main, joinpath(thisdir, "streams.jl"))
    Base.include(Main, joinpath(thisdir, "visualize.jl"))
end

"""
    load_config() -> Dict{String,Any}

Load configuration from `config.toml` located in this directory, with sane defaults.
Environment variables can override:
 - IDS_MAKE_GIF, IDS_GIF_STRIDE, IDS_RT_GIF_STRIDE
"""
function load_config()
    cfgpath = joinpath(@__DIR__, "config.toml")
    cfg = isfile(cfgpath) ? TOML.parsefile(cfgpath) : Dict{String,Any}()
    # Defaults
    cfg["initial_state"] = get(cfg, "initial_state", 0.0)
    cfg["observation_precision"] = get(cfg, "observation_precision", 0.1)
    cfg["seed"] = get(cfg, "seed", 123)
    cfg["n"] = get(cfg, "n", 300)
    cfg["interval_ms"] = get(cfg, "interval_ms", 41)
    cfg["iterations"] = get(cfg, "iterations", 10)
    cfg["rt_iterations"] = get(cfg, "rt_iterations", cfg["iterations"])
    cfg["keephistory"] = get(cfg, "keephistory", 10_000)
    cfg["output_dir"] = get(cfg, "output_dir", "output")
    cfg["fe_log_stride"] = get(cfg, "fe_log_stride", 10)
    # ENV overrides
    cfg["make_gif"] = get(ENV, "IDS_MAKE_GIF", get(cfg, "make_gif", true) ? "1" : "0") in ("1", "true", "TRUE")
    cfg["gif_stride"] = parse(Int, get(ENV, "IDS_GIF_STRIDE", string(get(cfg, "gif_stride", 5))))
    cfg["rt_gif_stride"] = parse(Int, get(ENV, "IDS_RT_GIF_STRIDE", string(get(cfg, "rt_gif_stride", 5))))
    cfg["skip_fe_replot"] = get(ENV, "IDS_SKIP_FE_REPLOT", "0") in ("1","true","TRUE")
    # Optional test-speed overrides
    if haskey(ENV, "IDS_N")
        try cfg["n"] = parse(Int, ENV["IDS_N"]) catch; end
    end
    if haskey(ENV, "IDS_INTERVAL_MS")
        try cfg["interval_ms"] = parse(Int, ENV["IDS_INTERVAL_MS"]) catch; end
    end
    if haskey(ENV, "IDS_ITERATIONS")
        try cfg["iterations"] = parse(Int, ENV["IDS_ITERATIONS"]) catch; end
    end
    if haskey(ENV, "IDS_SEED")
        try cfg["seed"] = parse(Int, ENV["IDS_SEED"]) catch; end
    end
    return cfg
end

end # module

