using Test

include(joinpath(@__DIR__, "..", "utils.jl")); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using .InfiniteDataStreamEnv

@testset "environment" begin
    env = Environment(0.0, 0.1; seed=123)
    @test env.current_state == 0.0
    @test length(gethistory(env)) == 0
    @test length(getobservations(env)) == 0
    y1 = getnext!(env)
    @test isa(y1, Float64)
    @test length(gethistory(env)) == 1
    @test length(getobservations(env)) == 1
end


