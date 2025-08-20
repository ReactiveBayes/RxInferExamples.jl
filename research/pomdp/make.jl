#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        if occursin("=", a)
            k, v = split(a, "=", limit=2)
            opts[k] = v
        else
            opts[a] = "true"
        end
    end
    return opts
end

opts = parse_args(ARGS)
run_mode = get(opts, "--mode", "full")  # quick | full | anim
nexp = get(opts, "--n", nothing)
hor  = get(opts, "--T", nothing)

if nexp !== nothing
    ENV["POMDP_N_EXP"] = nexp
end
if hor !== nothing
    ENV["POMDP_HOR"] = hor
end

if run_mode == "quick"
    include(joinpath(@__DIR__, "quick_run.jl"))
elseif run_mode == "full"
    include(joinpath(@__DIR__, "run_and_save_outputs.jl"))
elseif run_mode == "anim"
    include(joinpath(@__DIR__, "run_with_animation.jl"))
else
    println("Unknown mode: $run_mode. Use --mode=quick|full|anim")
end


