using Test
include(joinpath(@__DIR__, "..", "utils.jl")); using .InfiniteDataStreamUtils; InfiniteDataStreamUtils.load_modules!()
using .InfiniteDataStreamModel
using .InfiniteDataStreamUpdates
using .InfiniteDataStreamStreams

@testset "model and updates" begin
    au = make_autoupdates()
    init = make_initialization()
    @test !isnothing(au)
    @test !isnothing(init)
    # simple smoke: build infer with tiny stream
    using RxInfer
    data = [0.1, 0.2, 0.3]
    ds = to_namedtuple_stream(data)
    eng = infer(
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = ds,
        autoupdates    = au,
        returnvars     = (:x_current, ),
        keephistory    = 100,
        historyvars    = (x_current = KeepLast(),),
        initialization = init,
        iterations     = 2,
        autostart      = true,
        free_energy    = true,
    )
    @test eng.history !== nothing && haskey(eng.history, :x_current)
    @test !isempty(eng.free_energy_history)
end


