using Test

include(joinpath(@__DIR__, "..", "HGF.jl"))
using .HGF

@testset "HGF synthetic pipeline" begin
    params = default_hgf_params()
    z, x, y = generate_data(params)
    @test length(z) == params.n
    @test length(x) == params.n
    @test length(y) == params.n

    filter_result = run_filter(y, params.real_k, params.real_w, params.z_variance, params.y_variance)
    @test haskey(filter_result.history, :z_next)
    @test haskey(filter_result.history, :x_next)

    smoothing_result = run_smoothing(y, params.z_variance, params.y_variance)
    @test haskey(smoothing_result.posteriors, :z)
    @test haskey(smoothing_result.posteriors, :x)
    @test haskey(smoothing_result.posteriors, :κ)
    @test haskey(smoothing_result.posteriors, :ω)

    # Smoke-save a tiny output to ensure output dir exists and is writable
    outdir = ensure_output_dir()
    isdir(outdir) || error("Output directory not created: " * outdir)
end


