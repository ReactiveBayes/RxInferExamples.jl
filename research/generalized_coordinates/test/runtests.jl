using Test
using Random, LinearAlgebra, StableRNGs, Distributions, RxInfer

# Resolve modules without requiring package precompile
function _resolve_modules()
    try
        push!(LOAD_PATH, @__DIR__, joinpath(@__DIR__, ".."))
        @eval using GeneralizedCoordinatesExamples
        return (
            GeneralizedCoordinatesExamples.GCUtils,
            GeneralizedCoordinatesExamples.GCModel,
        )
    catch
        include(joinpath(@__DIR__, "..", "src", "GCUtils.jl")); @eval using .GCUtils
        include(joinpath(@__DIR__, "..", "src", "GCModel.jl")); @eval using .GCModel
        return (GCUtils, GCModel)
    end
end

const (GCUtils, GCModel) = _resolve_modules()

@testset "GC Car 1D position-only" begin
    rng = StableRNG(1)
    n, dt = 200, 0.1
    σ_a, σ_obs_pos = 0.2, 0.5
    x_true, y = GCUtils.generate_gc_car_data(rng, n, dt; σ_a=σ_a, σ_obs_pos=σ_obs_pos, σ_obs_vel=NaN)

    A, _, Qd = GCUtils.constant_acceleration_ABQ(dt; σ_a=σ_a)
    Q = Matrix(Qd)
    B = [1.0 0.0 0.0]
    R = Matrix(Diagonal([σ_obs_pos^2]))

    x0_mean = Float64[0.0, 0.0, 0.0]
    x0_cov  = Matrix(Diagonal(fill(100.0, 3)))

    result = infer(
        model = GCModel.gc_car_model(A=A, B=B, Q=Q, R=R, x0_mean=x0_mean, x0_cov=x0_cov),
        data  = (y = y,),
        constraints = GCModel.make_constraints(),
        returnvars = (x = KeepLast(),),
        free_energy = true,
        options = (limit_stack_depth = 500,)
    )

    xmarginals = result.posteriors[:x]
    @test length(xmarginals) == n
    @test size(mean.(xmarginals)[1], 1) == 3

    μ_pos = getindex.(mean.(xmarginals), 1)
    true_pos = getindex.(x_true, 1)
    mse = mean((μ_pos .- true_pos).^2)
    @test mse < 10.0

    @test all(isfinite, result.free_energy)
    @test result.free_energy[end] <= maximum(result.free_energy[1:min(end,10)]) + 1e-6
end

@testset "GC Car pos+vel observation" begin
    rng = StableRNG(2)
    n, dt = 150, 0.1
    σ_a, σ_obs_pos, σ_obs_vel = 0.2, 0.4, 0.6
    x_true, y = GCUtils.generate_gc_car_data(rng, n, dt; σ_a=σ_a, σ_obs_pos=σ_obs_pos, σ_obs_vel=σ_obs_vel)

    A, _, Qd = GCUtils.constant_acceleration_ABQ(dt; σ_a=σ_a)
    Q = Matrix(Qd)
    B = [1.0 0.0 0.0; 0.0 1.0 0.0]
    R = Matrix(Diagonal([σ_obs_pos^2, σ_obs_vel^2]))

    x0_mean = Float64[0.0, 0.0, 0.0]
    x0_cov  = Matrix(Diagonal(fill(100.0, 3)))

    result = infer(
        model = GCModel.gc_car_model(A=A, B=B, Q=Q, R=R, x0_mean=x0_mean, x0_cov=x0_cov),
        data  = (y = y,),
        constraints = GCModel.make_constraints(),
        returnvars = (x = KeepLast(),),
        free_energy = true,
        options = (limit_stack_depth = 500,)
    )

    xmarginals = result.posteriors[:x]
    @test length(xmarginals) == n
    μ_vel = getindex.(mean.(xmarginals), 2)
    true_vel = getindex.(x_true, 2)
    mse_v = mean((μ_vel .- true_vel).^2)
    @test mse_v < 10.0

    @test all(isfinite, result.free_energy)
end

@testset "Run script outputs smoke test" begin
    # Execute run script within its project and verify outputs
    include(joinpath(@__DIR__, "..", "run_gc_car.jl"))
    outdir = joinpath(@__DIR__, "..", "outputs")
    @test isfile(joinpath(outdir, "rxinfer_free_energy.csv"))
    @test isfile(joinpath(outdir, "gc_posterior_summary.csv"))
    @test isfile(joinpath(outdir, "gc_free_energy_timeseries.csv"))
end
