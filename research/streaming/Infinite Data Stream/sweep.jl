#!/usr/bin/env julia

# Sweep runner for Infinite Data Stream experiments
# Usage examples:
#   julia --project=. sweep.jl --configs sweep.toml
#   julia --project=. sweep.jl --configs sweep.toml --parallel false

import TOML
using Dates, DelimitedFiles, Statistics
using Distributed

include("utils.jl"); using .InfiniteDataStreamUtils
InfiniteDataStreamUtils.load_modules!()

include("runner.jl"); using .InfiniteDataStreamRunner

function parse_cli_args(args)
    cfg = Dict{String,Any}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--") && i < length(args)
            key = String(Symbol(replace(args[i][3:end], '-' => '_')))
            val = args[i+1]
            cfg[key] = val
            i += 2
        else
            i += 1
        end
    end
    return cfg
end

function load_runs_from_toml(path::AbstractString)
    @assert isfile(path) "configs file not found: $(path)"
    tbl = TOML.parsefile(path)
    runs = get(tbl, "run", [])
    @assert runs isa Vector "configs file must contain an array of [run] tables"
    return runs
end

function merge_cfg(base::Dict{String,Any}, override::Dict)
    out = copy(base)
    for (k,v) in override
        out[String(k)] = v
    end
    return out
end

function ensure_runs_csv(path::AbstractString)
    if !isfile(path)
        open(path, "w") do io
            println(io, "timestamp,n,interval_ms,iterations,rt_iterations,rt_fe_every,seed,output_root,static_mae,static_mse,realtime_mae,realtime_mse,outdir_static,outdir_realtime")
        end
    end
end

function append_run_row(csvpath::AbstractString, cfg::Dict{String,Any}, s::RunArtifacts, r::RunArtifacts)
    upto = min(length(s.truth), length(s.mu), length(r.truth), length(r.mu))
    μs = s.mu[1:upto]; μr = r.mu[1:upto]
    ts = s.truth[1:upto]; tr = r.truth[1:upto]
    mse_s = mean((μs .- ts).^2); mse_r = mean((μr .- tr).^2)
    mae_s = mean(abs.(μs .- ts)); mae_r = mean(abs.(μr .- tr))
    ts_now = Dates.format(now(), "yyyymmdd_HHMMSS")
    open(csvpath, "a") do io
        println(io, join([
            ts_now,
            get(cfg, "n", ""),
            get(cfg, "interval_ms", ""),
            get(cfg, "iterations", ""),
            get(cfg, "rt_iterations", ""),
            get(cfg, "rt_fe_every", ""),
            get(cfg, "seed", ""),
            get(cfg, "output_dir", "output"),
            string(mae_s), string(mse_s), string(mae_r), string(mse_r),
            joinpath(s.outdir, "static"), joinpath(r.outdir, "realtime")
        ], ","))
    end
end

function run_one(run_cfg)::Tuple{RunArtifacts,RunArtifacts,Dict{String,Any}}
    base = InfiniteDataStreamUtils.load_config()
    cfg = merge_cfg(base, run_cfg)
    # Ensure a shared timestamp for static and realtime so outputs collocate
    if !haskey(cfg, "ts")
        cfg["ts"] = Dates.format(now(), "yyyymmdd_HHMMSS")
    end
    s = run_static(cfg)
    r = run_realtime(cfg)
    compare_runs(s, r; outdir=joinpath(s.outdir, "comparison"))
    return (s, r, cfg)
end

function main()
    args = parse_cli_args(ARGS)
    cfg_path = get(args, "configs", "")
    parallel = lowercase(get(args, "parallel", "false")) in ("1","true","t","yes","y")
    runs = cfg_path == "" ? [] : load_runs_from_toml(cfg_path)
    @assert !isempty(runs) "No runs specified. Provide --configs sweep.toml with [run] tables."
    outcsv = joinpath(get(InfiniteDataStreamUtils.load_config(), "output_dir", "output"), "runs.csv")
    ensure_runs_csv(outcsv)
    if parallel
        try
            nprocs() == 1 && addprocs()
            # Initialize workers without @everywhere to avoid macro at non-toplevel
            for w in workers()
                remotecall_wait(include, w, "utils.jl")
                remotecall_eval(w, Main, :(using .InfiniteDataStreamUtils))
                remotecall_eval(w, Main, :(InfiniteDataStreamUtils.load_modules!()))
                remotecall_wait(include, w, "runner.jl")
                remotecall_eval(w, Main, :(using .InfiniteDataStreamRunner))
            end
            results = pmap(run_one, runs)
            for (s,r,cfg) in results
                append_run_row(outcsv, cfg, s, r)
            end
        catch
            println("[warn] parallel sweep failed to initialize; falling back to sequential")
            for run_cfg in runs
                s,r,cfg = run_one(run_cfg)
                append_run_row(outcsv, cfg, s, r)
            end
        end
    else
        for run_cfg in runs
            s,r,cfg = run_one(run_cfg)
            append_run_row(outcsv, cfg, s, r)
        end
    end
    println("Sweep complete. Results appended to $(outcsv)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


