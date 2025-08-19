using Test
include(joinpath(@__DIR__, "..", "utils.jl")); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using .InfiniteDataStreamViz

@testset "visualize" begin
    μ = collect(1.0:10.0)
    σ2 = fill(0.5, 10)
    hist = sin.(range(0, step=0.1, length=10)) .* 10
    obs = hist .+ 0.1
    p = plot_estimates(μ, σ2, hist, obs; upto=10)
    @test !isnothing(p)
    anim = animate_estimates(μ, σ2, hist, obs; stride=5)
    @test !isnothing(anim)
    anim_fe = animate_free_energy(collect(1.0:10.0); stride=5)
    @test !isnothing(anim_fe)
end


