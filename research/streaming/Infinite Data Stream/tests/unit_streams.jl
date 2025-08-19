using Test
include(joinpath(@__DIR__, "..", "utils.jl")); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using .InfiniteDataStreamStreams

@testset "streams" begin
    xs = [1.0, 2.0, 3.0]
    s = to_namedtuple_stream(xs)
    @test !isnothing(s)
    # smoke test for timer_observations: create small stream
    prod = let v = 0.0
        () -> (v += 1.0)
    end
    t = timer_observations(1, 3, prod)
    @test !isnothing(t)
end


