#!/usr/bin/env julia

# Activate the n-order GC project which includes RxInfer, StableRNGs, GraphViz
import Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using RxInfer
using GraphViz
using LinearAlgebra

include(abspath(joinpath(@__DIR__, "..", "src", "GeneralizedCoordinatesExamples.jl")))
using .GeneralizedCoordinatesExamples
const GCUtils = GeneralizedCoordinatesExamples.GCUtils
const GCModel = GeneralizedCoordinatesExamples.GCModel

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
render_gc_graphs_ffg(; orders=1:8, T=3, dt=0.1, observe_velocity=false, target_dir=abspath(@__DIR__))

For each order K, builds a minimal RxInfer factor graph for `gc_car_model` with horizon T
and saves a GraphViz-rendered FFG PNG into `target_dir`. This reflects the dimensionality
of the generalized coordinates (K) in the graph structure.
"""
function render_gc_graphs_ffg(; orders=1:8, T::Int=3, dt::Real=0.1,
                               observe_velocity::Bool=false,
                               target_dir::AbstractString=abspath(joinpath(@__DIR__, "images")))
    ensure_dir(target_dir)

    for K in orders
        # Build model parameters
        A, _, Qd = GCUtils.constant_acceleration_ABQ(dt; order=K, σ_a=0.5)
        Q = Matrix(Qd)
        if observe_velocity
            B = zeros(Float64, 2, K); B[1,1] = 1.0; B[2,2] = 1.0
            R = Matrix(Diagonal([0.5^2, 0.5^2]))
        else
            B = zeros(Float64, 1, K); B[1,1] = 1.0
            R = Matrix(Diagonal([0.5^2]))
        end
        x0_mean = zeros(K)
        x0_cov  = Matrix(Diagonal(fill(10.0, K)))

        # Minimal dummy y with correct dimension to define horizon/plate
        ny = size(B, 1)
        y = [zeros(ny) for _ in 1:T]

        # Create model instance and load FFG
        gen = GCModel.gc_car_model(y=y, A=A, B=B, Q=Q, R=R, x0_mean=x0_mean, x0_cov=x0_cov)
        model = RxInfer.getmodel(RxInfer.create_model(gen))

        # Export DOT and PNG via GraphViz
        g = GraphViz.load(model, strategy=:simple)
        base = "gc_car_model_ffg_order_$(K)"
        dot_path = joinpath(target_dir, base * ".dot")
        png_path = joinpath(target_dir, base * ".png")
        open(dot_path, "w") do io
            print(io, string(g))
        end
        try
            run(`dot -Tpng $(dot_path) -o $(png_path)`) # requires Graphviz
        catch
            @warn "Graphviz 'dot' not found; could not render $(png_path)."
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    render_gc_graphs_ffg()
end


