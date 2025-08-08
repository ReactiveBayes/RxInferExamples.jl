#!/usr/bin/env julia

using Printf

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
build_scalar_ffg_dot(K; T=3)

Construct a simple scalar-expanded DOT depiction of the generalized-coordinates
linear-Gaussian state-space model with order K and horizon T. Each scalar state
component x[t][k] is a node; prior, dynamics, and observation are factor nodes.
"""
function build_scalar_ffg_dot(K::Int; T::Int=3)
    io = IOBuffer()
    println(io, "digraph G {")
    println(io, "  rankdir=LR;")

    # Variable nodes x_t_k and y_t_d
    for t in 1:T
        println(io, @sprintf("  subgraph cluster_t%d { label=\"t=%d\"; style=dashed;", t, t))
        for k in 1:K
            println(io, @sprintf("    x_%d_%d [label=\"x[%d][%d]\", shape=circle];", t, k, t, k))
        end
        # Observation dims: position only
        println(io, @sprintf("    y_%d_1 [label=\"y[%d][1]\", shape=doublecircle];", t, t))
        println(io, "  }")
    end

    # Factor nodes: prior, dynamics, observation per time
    println(io, "  f_prior [label=\"Prior\", shape=box];")
    for t in 2:T
        println(io, @sprintf("  f_dyn_%d [label=\"Dynamics t=%d\", shape=box];", t, t))
    end
    for t in 1:T
        println(io, @sprintf("  f_obs_%d [label=\"Obs t=%d\", shape=box];", t, t))
    end

    # Edges: prior -> x_1_k; dynamics connects x_{t-1}_* to x_t_*; observation connects x_t_* to y_t
    for k in 1:K
        println(io, @sprintf("  f_prior -> x_1_%d;", k))
    end
    for t in 2:T
        for k in 1:K
            println(io, @sprintf("  x_%d_%d -> f_dyn_%d;", t-1, k, t))
            println(io, @sprintf("  f_dyn_%d -> x_%d_%d;", t, t, k))
        end
    end
    for t in 1:T
        for k in 1:K
            println(io, @sprintf("  x_%d_%d -> f_obs_%d;", t, k, t))
        end
        println(io, @sprintf("  f_obs_%d -> y_%d_1;", t, t))
    end

    println(io, "}")
    return String(take!(io))
end

function render_scalar_ffg_pngs(; orders=1:8, target_dir::AbstractString=abspath(joinpath(@__DIR__, "images")), T::Int=3)
    ensure_dir(target_dir)
    for K in orders
        dot = build_scalar_ffg_dot(K; T=T)
        dot_path = joinpath(target_dir, @sprintf("gc_car_model_scalar_order_%d.dot", K))
        png_path = joinpath(target_dir, @sprintf("gc_car_model_scalar_order_%d.png", K))
        open(dot_path, "w") do io; write(io, dot); end
        try
            run(`dot -Tpng $(dot_path) -o $(png_path)`)  # requires Graphviz
        catch
            @warn "Graphviz 'dot' not found; could not render $(png_path)."
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    render_scalar_ffg_pngs()
end


